"""
f5_trackerv2.py
===============
Daily F5 slate generator using pitcher-first model with offensive context.

Formula:
    xRUNS = pitcher_xRA * (opp_batting_xRuns / LEAGUE_AVG) ^ ALPHA * F5_SCALE
    Win%  = NegBin P(team_runs > opp_runs) + 0.5 * P(tie)   [r=6.0, max 15]
    FV    = American odds from Win%

Runs via cron each morning. Outputs:
    f5_slate_YYYY-MM-DD.xlsx   — formatted Excel slate
    f5_slate_YYYY-MM-DD.csv    — flat CSV for downstream use

Usage:
    python f5_trackerv2.py                  # today's slate
    python f5_trackerv2.py --date 2026-04-01  # specific date
    python f5_trackerv2.py --date 2026-04-01 --no-excel  # CSV only
"""

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from scipy.special import gammaln
from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment,
                              Border, Side, GradientFill)
from openpyxl.utils import get_column_letter

# ── CONFIG ─────────────────────────────────────────────────────────────────
DATA_DIR    = Path(".")          # where your CSV data files live
OUTPUT_DIR  = Path(".")          # where output files go

# Model constants (backtested against actual F5 outcomes 2023-2025)
ALPHA       = 1.2                # offensive context exponent
F5_SCALE    = 0.80               # calibrated run scaling factor
NEGBIN_R    = 6.0                # NegBin dispersion parameter
POISSON_MAX = 15                 # max runs in distribution table
LEAGUE_AVG  = 2.52               # league avg F5 runs (update seasonally)
MIN_STARTS  = 3                  # min career starts to use pitcher's xRA

# MLB Stats API
MLB_API     = "https://statsapi.mlb.com/api/v1"

# Team abbreviation map: MLB API codes → your tracker codes
MLB_TO_TRACKER = {
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BOS":"BOS","CHC":"CHC",
    "CWS":"CWS","CIN":"CIN","CLE":"CLE","COL":"COL","DET":"DET",
    "HOU":"HOU","KC" :"KCR","LAA":"LAA","LAD":"LAD","MIA":"MIA",
    "MIL":"MIL","MIN":"MIN","NYM":"NYM","NYY":"NYY","OAK":"ATH",
    "PHI":"PHI","PIT":"PIT","SD" :"SDP","SEA":"SEA","SF" :"SFG",
    "STL":"STL","TB" :"TBR","TEX":"TEX","TOR":"TOR","WSH":"WSN",
    # extras
    "ATH":"ATH","KCR":"KCR","SDP":"SDP","SFG":"SFG","TBR":"TBR","WSN":"WSN",
}

# ── NEGBIN WIN% ─────────────────────────────────────────────────────────────
def nb_pmf_mat(mu_arr: np.ndarray, r: float = NEGBIN_R,
               mx: int = POISSON_MAX) -> np.ndarray:
    """NegBin PMF matrix shape (n, mx+1) — vectorised."""
    k = np.arange(mx + 1)[None, :]
    p = r / (r + np.maximum(mu_arr, 0.01)[:, None])
    return np.exp(
        gammaln(k + r) - gammaln(r) - gammaln(k + 1)
        + r * np.log(p) + k * np.log(1 - p + 1e-15)
    )

def winpct_vec(mu_t: np.ndarray, mu_o: np.ndarray) -> np.ndarray:
    """Vectorised NegBin win%: P(team > opp) + 0.5*P(tie)."""
    T    = nb_pmf_mat(mu_t)
    O    = nb_pmf_mat(mu_o)
    Ocdf = np.cumsum(O, axis=1)
    win  = np.zeros(len(mu_t))
    tie  = np.zeros(len(mu_t))
    for a in range(POISSON_MAX + 1):
        cdf = Ocdf[:, a - 1] if a > 0 else np.zeros(len(mu_t))
        win += T[:, a] * cdf
        tie += T[:, a] * O[:, a]
    return win + 0.5 * tie

def prob_to_american(p: float) -> int:
    p = float(np.clip(p, 0.01, 0.99))
    return -round(p / (1 - p) * 100) if p >= 0.5 else round((1 - p) / p * 100)

# ── DATA LOADERS ─────────────────────────────────────────────────────────────
def load_pitcher_xra() -> dict:
    """Load career F5 RA per pitcher from f5_ra_by_pitcher.csv."""
    path = DATA_DIR / "f5_ra_by_pitcher.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[df["starts"] >= MIN_STARTS]
    return df.set_index(df["name"].str.strip())["f5_ra"].to_dict()

def load_batting_xruns() -> dict:
    """
    Compute season batting xRuns per team from f5_rs_tracker.csv.
    Returns dict: {(season, team): avg_f5_scored}
    Also returns current-season averages keyed by team alone.
    """
    path = DATA_DIR / "f5_rs_tracker.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["date"]   = pd.to_datetime(df["date"])
    df["team"]   = df["team"].str.strip().str.upper()
    df["season"] = df["date"].dt.year

    # Season averages
    season_avg = (df.groupby(["season", "team"])["f5_scored"]
                    .mean().reset_index()
                    .rename(columns={"f5_scored": "batxR"}))

    # Current season (most recent)
    cur_season = df["season"].max()
    cur = season_avg[season_avg["season"] == cur_season].set_index("team")["batxR"].to_dict()

    # Prior season as fallback
    prior_seasons = season_avg[season_avg["season"] < cur_season]
    prior = (prior_seasons.sort_values("season", ascending=False)
                          .drop_duplicates("team")
                          .set_index("team")["batxR"].to_dict())

    return cur, prior, df["f5_scored"].mean()

def load_park_factors() -> dict:
    """Load park factors keyed by home team."""
    path = DATA_DIR / "f5_park_factors.csv"
    if not path.exists():
        print("  WARNING: f5_park_factors.csv not found — park factors defaulting to 1.0")
        return {}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df.set_index(df["Team"].str.strip())["park_factor"].to_dict()

# ── MLB STARTERS SCRAPER ─────────────────────────────────────────────────────
def fetch_starters(game_date: date, retries: int = 3) -> list[dict]:
    """
    Fetch probable starters from MLB Stats API.
    Returns list of dicts with keys:
        game_id, game_time, away_team, home_team,
        away_pitcher, home_pitcher, venue
    """
    url = (f"{MLB_API}/schedule"
           f"?sportId=1&date={game_date.strftime('%Y-%m-%d')}"
           f"&hydrate=probablePitcher,teams,venue"
           f"&gameType=R")

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"MLB API failed after {retries} attempts: {e}")

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]
            home = g["teams"]["home"]

            # Normalize team codes
            away_code = MLB_TO_TRACKER.get(
                away["team"].get("abbreviation", ""), "UNK")
            home_code = MLB_TO_TRACKER.get(
                home["team"].get("abbreviation", ""), "UNK")

            away_p = away.get("probablePitcher", {})
            home_p = home.get("probablePitcher", {})

            # Parse game time (UTC → local approximate)
            gt = g.get("gameDate", "")
            try:
                game_time = datetime.fromisoformat(
                    gt.replace("Z", "+00:00")).strftime("%-I:%M %p")
            except Exception:
                game_time = "TBD"

            games.append({
                "game_id":      g.get("gamePk", ""),
                "game_time":    game_time,
                "away_team":    away_code,
                "home_team":    home_code,
                "away_name":    away["team"].get("name", away_code),
                "home_name":    home["team"].get("name", home_code),
                "away_pitcher": away_p.get("fullName", "TBD"),
                "home_pitcher": home_p.get("fullName", "TBD"),
                "venue":        g.get("venue", {}).get("name", ""),
            })

    return games

# ── MODEL ─────────────────────────────────────────────────────────────────────
def build_slate(games: list[dict],
                pitcher_xra: dict,
                bat_cur: dict,
                bat_prior: dict,
                league_bat: float,
                park_factors: dict) -> pd.DataFrame:
    """
    Compute xRUNS, Win%, FV for every game on the slate.
    """
    rows = []

    for g in games:
        away = g["away_team"]
        home = g["home_team"]

        # ── Pitcher xRA ──────────────────────────────────────────────────
        away_pxra = pitcher_xra.get(g["away_pitcher"],
                    pitcher_xra.get(g["away_pitcher"].split()[-1], LEAGUE_AVG))
        home_pxra = pitcher_xra.get(g["home_pitcher"],
                    pitcher_xra.get(g["home_pitcher"].split()[-1], LEAGUE_AVG))

        # Flag if xRA was not found (fallback to league avg)
        away_p_found = g["away_pitcher"] in pitcher_xra
        home_p_found = g["home_pitcher"] in pitcher_xra

        # ── Batting xRuns ────────────────────────────────────────────────
        away_bat = bat_cur.get(away, bat_prior.get(away, league_bat))
        home_bat = bat_cur.get(home, bat_prior.get(home, league_bat))

        # ── Park factor ──────────────────────────────────────────────────
        pf = park_factors.get(home, 1.0)

        # ── xRUNS (pitcher-first, offense-adjusted, park-adjusted) ──────
        # away xRUNS = runs away team scores = f(home pitcher, away batting)
        # home xRUNS = runs home team scores = f(away pitcher, home batting)
        away_xruns_raw = home_pxra * (away_bat / LEAGUE_AVG) ** ALPHA * F5_SCALE
        home_xruns_raw = away_pxra * (home_bat / LEAGUE_AVG) ** ALPHA * F5_SCALE

        # Apply park factor to both sides (park affects total, split evenly)
        away_xruns = away_xruns_raw * pf
        home_xruns = home_xruns_raw * pf

        # ── Win% and FV ──────────────────────────────────────────────────
        away_wp = float(winpct_vec(
            np.array([max(away_xruns, 0.1)]),
            np.array([max(home_xruns, 0.1)])
        )[0])
        home_wp = 1.0 - away_wp

        away_fv = prob_to_american(away_wp)
        home_fv = prob_to_american(home_wp)

        rows.append({
            "Game Time":        g["game_time"],
            "Away Team":        g["away_name"],
            "Away Code":        away,
            "Away Pitcher":     g["away_pitcher"],
            "Away P Found":     away_p_found,
            "Away pxRA":        round(away_pxra, 3),
            "Away Bat xR":      round(away_bat, 3),
            "Away xRUNS":       round(away_xruns, 3),
            "Away Win%":        round(away_wp * 100, 1),
            "Away FV":          away_fv,
            "Home Team":        g["home_name"],
            "Home Code":        home,
            "Home Pitcher":     g["home_pitcher"],
            "Home P Found":     home_p_found,
            "Home pxRA":        round(home_pxra, 3),
            "Home Bat xR":      round(home_bat, 3),
            "Home xRUNS":       round(home_xruns, 3),
            "Home Win%":        round(home_wp * 100, 1),
            "Home FV":          home_fv,
            "Park Factor":      round(pf, 4),
            "Venue":            g["venue"],
        })

    return pd.DataFrame(rows)

# ── EXCEL OUTPUT ──────────────────────────────────────────────────────────────
def write_excel(df: pd.DataFrame, game_date: date, path: Path) -> None:
    """Write formatted Excel slate."""
    wb = Workbook()
    ws = wb.active
    ws.title = "F5 Slate"

    # ── Colour palette ───────────────────────────────────────────────────
    CLR_HEADER   = "1F3864"   # dark navy
    CLR_AWAY     = "D6E4F0"   # light blue
    CLR_HOME     = "D5F5E3"   # light green
    CLR_WARN     = "FFF3CD"   # amber — pitcher not found
    CLR_TITLE    = "2E75B6"
    WHITE        = "FFFFFF"
    DARK_TEXT    = "FFFFFF"
    BLACK        = "000000"

    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def hdr_fill(hex_c):
        return PatternFill("solid", start_color=hex_c, fgColor=hex_c)

    def cell_fill(hex_c):
        return PatternFill("solid", start_color=hex_c, fgColor=hex_c)

    # ── Title row ────────────────────────────────────────────────────────
    ws.merge_cells("A1:T1")
    tc = ws["A1"]
    tc.value        = f"ClutchSheets — F5 Innings Slate  |  {game_date.strftime('%B %d, %Y')}"
    tc.font         = Font(name="Arial", bold=True, size=14, color=WHITE)
    tc.fill         = hdr_fill(CLR_TITLE)
    tc.alignment    = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    # ── Sub-header row ───────────────────────────────────────────────────
    ws["A2"] = f"Model: pitcher xRA × (opp_bat / {LEAGUE_AVG})^{ALPHA} × {F5_SCALE}  |  NegBin r={NEGBIN_R}"
    ws["A2"].font = Font(name="Arial", italic=True, size=9, color="595959")
    ws.merge_cells("A2:T2")
    ws["A2"].alignment = Alignment(horizontal="left")
    ws.row_dimensions[2].height = 14

    # ── Column headers ───────────────────────────────────────────────────
    COLS = [
        ("Time",        "Game Time",   9),
        ("Away",        "Away Team",   18),
        ("Away P",      "Away Pitcher",20),
        ("AxRA",        "Away pxRA",   9),
        ("ABat",        "Away Bat xR", 9),
        ("AxR",         "Away xRUNS",  9),
        ("AW%",         "Away Win%",   8),
        ("AFV",         "Away FV",     8),
        ("vs",          "",            3),
        ("Home",        "Home Team",   18),
        ("Home P",      "Home Pitcher",20),
        ("HxRA",        "Home pxRA",   9),
        ("HBat",        "Home Bat xR", 9),
        ("HxR",         "Home xRUNS",  9),
        ("HW%",         "Home Win%",   8),
        ("HFV",         "Home FV",     8),
        ("PF",          "Park Factor", 9),
        ("Venue",       "Venue",       22),
    ]

    HDR_ROW = 3
    df_cols = [
        "Game Time","Away Team","Away Pitcher","Away pxRA","Away Bat xR",
        "Away xRUNS","Away Win%","Away FV","",
        "Home Team","Home Pitcher","Home pxRA","Home Bat xR",
        "Home xRUNS","Home Win%","Home FV","Park Factor","Venue",
    ]

    for col_idx, (_, label, width) in enumerate(COLS, start=1):
        c = ws.cell(row=HDR_ROW, column=col_idx, value=label)
        c.font      = Font(name="Arial", bold=True, size=10, color=WHITE)
        c.fill      = hdr_fill(CLR_HEADER)
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border    = border
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.row_dimensions[HDR_ROW].height = 18

    # ── Data rows ────────────────────────────────────────────────────────
    START_ROW = 4
    for r_idx, row in df.iterrows():
        excel_row = START_ROW + r_idx

        # Determine fills
        away_fill = cell_fill(CLR_AWAY if row["Away P Found"] else CLR_WARN)
        home_fill = cell_fill(CLR_HOME if row["Home P Found"] else CLR_WARN)

        row_vals = [
            row["Game Time"],
            row["Away Team"],
            row["Away Pitcher"],
            row["Away pxRA"],
            row["Away Bat xR"],
            row["Away xRUNS"],
            f"{row['Away Win%']}%",
            f"+{row['Away FV']}" if row["Away FV"] > 0 else str(row["Away FV"]),
            "vs",
            row["Home Team"],
            row["Home Pitcher"],
            row["Home pxRA"],
            row["Home Bat xR"],
            row["Home xRUNS"],
            f"{row['Home Win%']}%",
            f"+{row['Home FV']}" if row["Home FV"] > 0 else str(row["Home FV"]),
            row["Park Factor"],
            row["Venue"],
        ]

        for col_idx, val in enumerate(row_vals, start=1):
            c = ws.cell(row=excel_row, column=col_idx, value=val)
            c.font      = Font(name="Arial", size=10)
            c.border    = border
            c.alignment = Alignment(horizontal="center")

            # Away side styling (cols 1-8)
            if col_idx <= 8:
                c.fill = away_fill
                if col_idx == 8:  # FV
                    c.font = Font(name="Arial", size=10, bold=True,
                                  color="1A5276" if row["Away Win%"] >= 50 else BLACK)
            # "vs" column
            elif col_idx == 9:
                c.font = Font(name="Arial", size=9, color="888888")
                c.fill = cell_fill("F8F9FA")
            # Home side styling (cols 10-16)
            elif col_idx <= 16:
                c.fill = home_fill
                if col_idx == 16:  # FV
                    c.font = Font(name="Arial", size=10, bold=True,
                                  color="1A5276" if row["Home Win%"] >= 50 else BLACK)
            # Park factor + venue
            else:
                c.fill = cell_fill("F8F9FA")
                if col_idx == 17 and row["Park Factor"] > 1.05:
                    c.font = Font(name="Arial", size=10, bold=True, color="C0392B")
                elif col_idx == 17 and row["Park Factor"] < 0.95:
                    c.font = Font(name="Arial", size=10, bold=True, color="1A5276")

        ws.row_dimensions[excel_row].height = 16

    # ── Legend ───────────────────────────────────────────────────────────
    legend_row = START_ROW + len(df) + 2
    ws.cell(row=legend_row, column=1,
            value="★ Amber shading = pitcher not found in xRA database (league avg used)").font \
        = Font(name="Arial", size=9, italic=True, color="7F6000")
    ws.cell(row=legend_row+1, column=1,
            value=f"★ Park factor > 1.05 = hitter friendly (red)  |  < 0.95 = pitcher friendly (blue)").font \
        = Font(name="Arial", size=9, italic=True, color="595959")

    # ── Freeze panes ─────────────────────────────────────────────────────
    ws.freeze_panes = ws.cell(row=START_ROW, column=1)

    wb.save(path)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="F5 daily slate generator")
    parser.add_argument("--date", default=None,
                        help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-excel", action="store_true",
                        help="Skip Excel output, CSV only")
    args = parser.parse_args()

    game_date = (date.today() if args.date is None
                 else datetime.strptime(args.date, "%Y-%m-%d").date())

    print(f"F5 Tracker v2  |  {game_date}")
    print("─" * 48)

    # Load data
    print("Loading data...")
    pitcher_xra          = load_pitcher_xra()
    bat_cur, bat_prior, league_bat = load_batting_xruns()
    park_factors         = load_park_factors()
    print(f"  Pitchers in xRA db: {len(pitcher_xra)}")
    print(f"  Teams with batting xRuns (current): {len(bat_cur)}")
    print(f"  League avg F5 runs: {league_bat:.3f}")
    print(f"  Park factors: {len(park_factors)} teams")

    # Fetch starters
    print(f"\nFetching probable starters for {game_date}...")
    games = fetch_starters(game_date)
    print(f"  Games found: {len(games)}")
    if not games:
        print("  No games found for this date. Exiting.")
        sys.exit(0)

    # Pitch lookup report
    all_pitchers = ([g["away_pitcher"] for g in games]
                  + [g["home_pitcher"] for g in games])
    found    = sum(1 for p in all_pitchers if p in pitcher_xra)
    missing  = [p for p in all_pitchers if p != "TBD" and p not in pitcher_xra]
    print(f"  Pitchers found in xRA db: {found}/{len(all_pitchers)}")
    if missing:
        print(f"  Missing (league avg used): {', '.join(missing)}")

    # Build slate
    print("\nComputing projections...")
    df = build_slate(games, pitcher_xra, bat_cur, bat_prior,
                     league_bat, park_factors)

    # Print to console
    print("\n" + "=" * 70)
    print(f"{'GAME':<35} {'AWAY FV':>8} {'AWAY W%':>8} {'HOME FV':>8} {'HOME W%':>8}")
    print("=" * 70)
    for _, row in df.iterrows():
        game_str = f"{row['Away Team']} @ {row['Home Team']}"
        afv = f"+{row['Away FV']}" if row['Away FV'] > 0 else str(row['Away FV'])
        hfv = f"+{row['Home FV']}" if row['Home FV'] > 0 else str(row['Home FV'])
        print(f"{game_str:<35} {afv:>8} {row['Away Win%']:>7}% {hfv:>8} {row['Home Win%']:>7}%")
    print("=" * 70)

    # Save CSV (clean columns)
    csv_cols = ["Game Time","Away Team","Away Pitcher","Away pxRA",
                "Away Bat xR","Away xRUNS","Away Win%","Away FV",
                "Home Team","Home Pitcher","Home pxRA","Home Bat xR",
                "Home xRUNS","Home Win%","Home FV","Park Factor","Venue"]
    date_str  = game_date.strftime("%Y-%m-%d")
    csv_path  = OUTPUT_DIR / f"f5_slate_{date_str}.csv"
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    # Save Excel
    if not args.no_excel:
        xlsx_path = OUTPUT_DIR / f"f5_slate_{date_str}.xlsx"
        write_excel(df, game_date, xlsx_path)
        print(f"Excel saved: {xlsx_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
