"""
f5_pipeline.py  — ClutchSheets F5 Daily Pipeline  v3.0
=======================================================
Single script that does everything in order:
  1. Rebuilds f5_ra_by_pitcher.csv using hard-window xRA (last 8–10 starts)
  2. Fetches today's confirmed/probable starters from MLB Stats API
  3. Enriches each game with umpire zone tendency, pitcher days-rest flags,
     and actual lineup L/R composition vs pitcher hand
  4. Computes xRUNS / Win% / FV using pitcher-first NegBin model
  5. Outputs Excel slate + two CSVs

Formula:
    xRUNS = pitcher_xRA(window) * (opp_bat_hand_adj / LEAGUE_AVG)^1.2
            * lineup_factor * 0.80 * ump_factor * rest_factor * park_factor
    Win%  = NegBin P(team > opp) + 0.5*P(tie)   [r=6.0, max 15 runs]
    FV    = American odds from Win%

New in v3.0:
  • XRA_WINDOW (8–10 starts hard cap) — decay over window only, no full history
  • Confirmed starter check — flags games still showing probable vs confirmed
  • Umpire home-plate zone tendency — squeeze (<−0.02 adj) / expand (>+0.02 adj)
  • Pitcher days-rest flags — short rest (≤3d) and IL-return penalties
  • Lineup handedness — actual L/R count from pre-game lineup vs pitcher hand

Usage:
    python f5_pipeline.py                    # today
    python f5_pipeline.py --date 2026-04-01  # specific date
    python f5_pipeline.py --no-excel         # skip Excel, CSVs only

Outputs (D:\\Desktop\\F5 Baseball Forecaster\\Model\\):
    f5_ra_by_pitcher.csv          updated pitcher xRA
    f5_slate_YYYY-MM-DD.xlsx      formatted Excel slate
    f5_slate_YYYY-MM-DD.csv       full detail CSV

Output (D:\\Desktop\\F5 Baseball Forecaster\\):
    f5_forecast_YYYY-MM-DD.csv    pipeline-format CSV (lambda/winprob)
"""

import argparse, subprocess, sys, time, unicodedata
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy.special import gammaln

# ── PATHS ──────────────────────────────────────────────────────────────────
MODEL_DIR    = Path(r"D:\Desktop\F5 Baseball Forecaster\Model")  # tracker CSVs + scripts
FORECAST_DIR = Path(r"D:\Desktop\F5 Baseball Forecaster")         # repo root — daily_forecast.csv lives here

# ── MODEL CONSTANTS ────────────────────────────────────────────────────────
ALPHA        = 1.2     # offensive context exponent (backtested)
F5_SCALE     = 0.80    # run scaling factor (backtested)
NEGBIN_R     = 6.0     # NegBin dispersion parameter
POISSON_MAX  = 15      # max runs in distribution table
LEAGUE_AVG   = 2.52    # league avg F5 runs — update mid-season if needed
MIN_STARTS   = 5       # min starts to include a pitcher in xRA db

# ── xRA WINDOW CONSTANTS (v3.0 — hard window replaces full-history decay) ──
XRA_WINDOW   = 10      # max starts used in xRA calculation (hard cap)
XRA_WINDOW_MIN = 8     # starts below this triggers career-blend
DECAY        = 0.85    # per-start decay within window
BLEND_CUTOFF = 8       # starts needed for full recency confidence
CAREER_BLEND = 0.35    # career weight blended in when starts < BLEND_CUTOFF

# ── v3.0 ENRICHMENT ADJUSTMENTS ────────────────────────────────────────────
# Umpire zone: runs-per-game adjustment (expand = +, squeeze = -)
UMP_EXPAND_THRESH  =  0.20   # ump_run_delta above this → expansion zone
UMP_SQUEEZE_THRESH =  0.10   # ump_run_delta below this → squeeze zone
UMP_EXPAND_MULT    =  1.04   # xRUNS multiplier in expansion zone
UMP_SQUEEZE_MULT   =  0.96   # xRUNS multiplier in squeeze zone

# Days-rest penalties
SHORT_REST_DAYS    = 3       # ≤ this many days = short rest
SHORT_REST_PENALTY = 1.12    # multiply pitcher xRA by this on short rest
IL_RETURN_PENALTY  = 1.08    # multiply pitcher xRA by this on IL return

# Lineup handedness: multiplier applied to batting xR
# derived from historical platoon advantage (vs LHP / vs RHP)
LH_PLATOON_BONUS   = 1.06   # team lined up favourably (more same-hand batters vs pitcher)
RH_PLATOON_BONUS   = 1.00   # neutral baseline
PLATOON_THRESHOLD  = 0.55   # fraction of same-hand batters to trigger bonus

# ── MLB STATS API ──────────────────────────────────────────────────────────
MLB_API = "https://statsapi.mlb.com/api/v1"

MLB_TO_TRACKER = {
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BOS":"BOS","CHC":"CHC",
    "CWS":"CWS","CIN":"CIN","CLE":"CLE","COL":"COL","DET":"DET",
    "HOU":"HOU","KC" :"KCR","LAA":"LAA","LAD":"LAD","MIA":"MIA",
    "MIL":"MIL","MIN":"MIN","NYM":"NYM","NYY":"NYY","OAK":"ATH",
    "PHI":"PHI","PIT":"PIT","SD" :"SDP","SEA":"SEA","SF" :"SFG",
    "STL":"STL","TB" :"TBR","TEX":"TEX","TOR":"TOR","WSH":"WSN",
    "ATH":"ATH","KCR":"KCR","SDP":"SDP","SFG":"SFG","TBR":"TBR","WSN":"WSN",
    "TBA":"TBR","KCA":"KCR","SFN":"SFG","SDN":"SDP","NYN":"NYM",
    "NYA":"NYY","CHA":"CWS","CHN":"CHC","LAN":"LAD","ANA":"LAA",
    "WAS":"WSN","SLN":"STL","FLO":"MIA",
}

MLB_NAME_TO_TRACKER = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
    "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KCR",
    "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
    "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
    "New York Yankees":"NYY","Oakland Athletics":"ATH","Athletics":"ATH",
    "Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT","San Diego Padres":"SDP",
    "Seattle Mariners":"SEA","San Francisco Giants":"SFG","St. Louis Cardinals":"STL",
    "Tampa Bay Rays":"TBR","Texas Rangers":"TEX","Toronto Blue Jays":"TOR",
    "Washington Nationals":"WSN",
}

# ── HELPERS ────────────────────────────────────────────────────────────────
def norm_name(s: str) -> str:
    """Strip accents for pitcher name matching (Pérez → Perez)."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def prob_to_american(p: float) -> int:
    p = float(np.clip(p, 0.01, 0.99))
    return -round(p / (1 - p) * 100) if p >= 0.5 else round((1 - p) / p * 100)

# ── NEGBIN WIN% ────────────────────────────────────────────────────────────
def nb_pmf_mat(mu_arr: np.ndarray, r: float = NEGBIN_R, mx: int = POISSON_MAX):
    k = np.arange(mx + 1)[None, :]
    p = r / (r + np.maximum(mu_arr, 0.01)[:, None])
    return np.exp(gammaln(k+r) - gammaln(r) - gammaln(k+1)
                  + r*np.log(p) + k*np.log(1-p+1e-15))

def winpct_vec(mu_t: np.ndarray, mu_o: np.ndarray) -> np.ndarray:
    T = nb_pmf_mat(mu_t); O = nb_pmf_mat(mu_o); Ocdf = np.cumsum(O, axis=1)
    win = np.zeros(len(mu_t)); tie = np.zeros(len(mu_t))
    for a in range(POISSON_MAX + 1):
        cdf = Ocdf[:, a-1] if a > 0 else np.zeros(len(mu_t))
        win += T[:, a] * cdf; tie += T[:, a] * O[:, a]
    return win + 0.5 * tie

# ── STEP 1: REBUILD PITCHER xRA (hard window) ─────────────────────────────
def rebuild_pitcher_xra() -> None:
    """
    Recompute f5_ra_by_pitcher.csv from f5_ra_tracker.csv.

    v3.0 change: uses a hard window of the last XRA_WINDOW starts (default 10)
    rather than full history.  Decay is applied within that window only.
    Career blend still kicks in when a pitcher has fewer than BLEND_CUTOFF
    qualifying starts inside the window.
    """
    tracker_path = MODEL_DIR / "f5_ra_tracker.csv"
    output_path  = MODEL_DIR / "f5_ra_by_pitcher.csv"

    df = pd.read_csv(tracker_path)
    df.columns    = df.columns.str.strip()
    df["date"]    = pd.to_datetime(df["date"])
    df["pitcher"] = df["pitcher"].str.strip()

    counts = df.groupby("pitcher").size()
    df     = df[df["pitcher"].isin(counts[counts >= MIN_STARTS].index)]

    results = []
    for pitcher, grp in df.groupby("pitcher"):
        # ── hard window: keep only last XRA_WINDOW starts ──────────────────
        all_runs  = grp.sort_values("date", ascending=False)["f5_allowed"].values
        windowed  = all_runs[:XRA_WINDOW]          # hard cap
        n         = len(windowed)
        weights   = np.array([DECAY ** i for i in range(n)])
        xra_r     = float(np.average(windowed, weights=weights))

        # Career avg uses all history (not just window) for stability
        xra_c     = float(all_runs.mean())

        if n >= BLEND_CUTOFF:
            xra_f = xra_r
        else:
            cw    = CAREER_BLEND * (1 - (n - MIN_STARTS) / (BLEND_CUTOFF - MIN_STARTS))
            cw    = max(0.0, min(CAREER_BLEND, cw))
            xra_f = (1 - cw) * xra_r + cw * xra_c

        results.append({
            "name":         pitcher,
            "f5_ra":        round(xra_f, 4),
            "starts":       n,                     # window count
            "total_starts": len(all_runs),         # for reference
        })

    out = pd.DataFrame(results).sort_values("f5_ra").reset_index(drop=True)
    out.to_csv(output_path, index=False)
    print(f"  xRA rebuilt: {len(out)} pitchers  "
          f"window={XRA_WINDOW}  "
          f"range {out['f5_ra'].min():.3f}–{out['f5_ra'].max():.3f}")

# ── STEP 2: LOAD DATA ──────────────────────────────────────────────────────
def load_pitcher_xra() -> dict:
    path = MODEL_DIR / "f5_ra_by_pitcher.csv"
    df   = pd.read_csv(path); df.columns = df.columns.str.strip()
    df   = df[df["starts"] >= MIN_STARTS]
    xra  = df.set_index(df["name"].str.strip())["f5_ra"].to_dict()

    # Encode handedness from tracker
    tr_path = MODEL_DIR / "f5_ra_tracker.csv"
    if tr_path.exists():
        tr = pd.read_csv(tr_path); tr.columns = tr.columns.str.strip()
        tr["pitcher"] = tr["pitcher"].str.strip()
        tr["throws"]  = tr["throws"].str.strip().str.upper()
        hand = (tr.groupby("pitcher")["throws"]
                  .agg(lambda x: x.mode()[0] if len(x) else "R").to_dict())
        for name, h in hand.items():
            xra[f"__hand__{name}"] = h

    # Also store accent-normalised keys
    for name in list(xra.keys()):
        n2 = norm_name(name)
        if n2 != name:
            xra.setdefault(n2, xra[name])
            xra.setdefault(f"__hand__{n2}", xra.get(f"__hand__{name}", "R"))

    return xra

def load_batting_xruns():
    path = MODEL_DIR / "f5_rs_tracker.csv"
    df   = pd.read_csv(path); df.columns = df.columns.str.strip()
    df["date"]     = pd.to_datetime(df["date"])
    df["team"]     = df["team"].str.strip().str.upper()
    df["opp_hand"] = df["opp_hand"].str.strip().str.upper()
    df["season"]   = df["date"].dt.year
    df             = df.dropna(subset=["team"])

    counts  = df.groupby("season").size()
    cur_season = counts[counts >= 200].index.max()
    cur     = df[df["season"] == cur_season]

    split   = cur.groupby(["team","opp_hand"])["f5_scored"].mean().reset_index()
    bat_R   = split[split["opp_hand"]=="R"].set_index("team")["f5_scored"].to_dict()
    bat_L   = split[split["opp_hand"]=="L"].set_index("team")["f5_scored"].to_dict()
    bat_flat= cur.groupby("team")["f5_scored"].mean().to_dict()
    prior   = (df[df["season"] < cur_season]
                 .groupby(["season","team"])["f5_scored"].mean()
                 .reset_index().sort_values("season",ascending=False)
                 .drop_duplicates("team").set_index("team")["f5_scored"].to_dict())

    return bat_R, bat_L, bat_flat, prior, df["f5_scored"].mean()

def load_park_factors() -> dict:
    path = MODEL_DIR / "f5_park_factors.csv"
    if not path.exists():
        print("  WARNING: park factors not found — defaulting to 1.0"); return {}
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    return df.set_index(df["Team"].str.strip())["park_factor"].to_dict()

# ── STEP 3: FETCH STARTERS (v3.0) ─────────────────────────────────────────
def _mlb_get(url: str, params: dict = None, retries: int = 3) -> dict:
    """Shared retry-wrapper for MLB Stats API calls."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"MLB API failed [{url}]: {e}")


def fetch_umpire_factors(game_date: date) -> dict:
    """
    Return {game_pk: ump_factor} where ump_factor is an xRUNS multiplier.

    Hits the /schedule endpoint with officialScorer / umpires hydration.
    Falls back to 1.0 if the data isn't available (pre-game assignments often
    post ~3–4 hours before first pitch).

    The ump_run_delta in the umpire table is expected to be a pre-computed
    per-game runs-above-average figure.  If you don't yet have that table,
    this function defaults everything to 1.0 gracefully.
    """
    ump_ratings_path = MODEL_DIR / "f5_ump_ratings.csv"
    ump_ratings: dict = {}
    if ump_ratings_path.exists():
        udf = pd.read_csv(ump_ratings_path, encoding="latin-1")
        udf.columns = udf.columns.str.strip()
        if "umpire_name" in udf.columns and "ump_run_delta" in udf.columns:
            udf["umpire_name"] = udf["umpire_name"].str.strip()
            ump_ratings = udf.set_index("umpire_name")["ump_run_delta"].to_dict()

    url = (f"{MLB_API}/schedule?sportId=1"
           f"&date={game_date.strftime('%Y-%m-%d')}"
           f"&hydrate=officials&gameType=R")
    try:
        data = _mlb_get(url)
    except Exception as e:
        print(f"  [UMP] API call failed — defaulting to 1.0: {e}")
        return {}

    ump_factors: dict = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            pk = g.get("gamePk")
            officials = g.get("officials", [])
            hp_ump = next(
                (o.get("official", {}).get("fullName", "")
                 for o in officials
                 if o.get("officialType", "").lower() == "home plate"),
                None,
            )
            if not hp_ump or not ump_ratings:
                ump_factors[pk] = 1.0
                continue
            delta = ump_ratings.get(hp_ump, 0.0)
            if delta >= UMP_EXPAND_THRESH:
                ump_factors[pk] = UMP_EXPAND_MULT
            elif delta <= UMP_SQUEEZE_THRESH:
                ump_factors[pk] = UMP_SQUEEZE_MULT
            else:
                ump_factors[pk] = 1.0
            if hp_ump:
                ump_factors[f"__name__{pk}"] = hp_ump
    return ump_factors


def fetch_pitcher_rest(game_date: date, pitcher_id: int) -> dict:
    """
    Return {"days_rest": int, "il_return": bool} for a pitcher.

    Checks the pitcher's transaction log + recent game log via the API.
    days_rest is calendar days since their last appearance.
    il_return is True if they appear on an active roster after an IL stint
    in the past 30 days.
    """
    result = {"days_rest": 99, "il_return": False}
    if not pitcher_id:
        return result

    # Game log: last appearance date
    try:
        log = _mlb_get(
            f"{MLB_API}/people/{pitcher_id}/stats",
            params={"stats": "gameLog", "group": "pitching",
                    "season": game_date.year, "limit": 5},
        )
        games_logged = (log.get("stats", [{}])[0]
                           .get("splits", []) if log.get("stats") else [])
        if games_logged:
            last_date_str = games_logged[0].get("date", "")
            if last_date_str:
                last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                result["days_rest"] = (game_date - last_date).days
    except Exception:
        pass  # silently keep default 99

    # Transaction log: IL activity in last 30 days
    try:
        txn = _mlb_get(
            f"{MLB_API}/transactions",
            params={"playerId": pitcher_id,
                    "startDate": (game_date.replace(day=1)).strftime("%m/%d/%Y"),
                    "endDate":   game_date.strftime("%m/%d/%Y")},
        )
        for t in txn.get("transactions", []):
            desc = t.get("typeDesc", "").lower()
            if "reinstated" in desc or "activated" in desc:
                result["il_return"] = True
                break
    except Exception:
        pass

    return result


def fetch_lineup_handedness(game_pk: int, away_pitcher_hand: str,
                            home_pitcher_hand: str) -> dict:
    """
    Return {"away_lineup_factor": float, "home_lineup_factor": float}
    based on actual pre-game lineup L/R composition vs opposing pitcher hand.

    If the lineup isn't posted yet, returns 1.0 for both sides.
    """
    result = {"away_lineup_factor": 1.0, "home_lineup_factor": 1.0}
    if not game_pk:
        return result

    try:
        feed = _mlb_get(f"{MLB_API}/game/{game_pk}/boxscore")
        teams = feed.get("teams", {})
        for side, sp_hand in (("away", home_pitcher_hand),
                               ("home", away_pitcher_hand)):
            batters = teams.get(side, {}).get("battingOrder", [])
            if not batters:
                continue
            # Pull batter handedness from the players dict
            players = teams.get(side, {}).get("players", {})
            hands = []
            for pid in batters:
                key = f"ID{pid}"
                bat_side = (players.get(key, {})
                                   .get("batSide", {})
                                   .get("code", "R"))
                hands.append(bat_side)
            if not hands:
                continue
            n = len(hands)
            # Count batters who have platoon advantage vs this pitcher hand
            # LH pitcher → R batters have advantage; RH pitcher → L batters have advantage
            adv_hand  = "R" if sp_hand == "L" else "L"
            adv_count = sum(1 for h in hands if h == adv_hand or h == "S")
            frac_adv  = adv_count / n
            factor    = LH_PLATOON_BONUS if frac_adv >= PLATOON_THRESHOLD else RH_PLATOON_BONUS
            result[f"{side}_lineup_factor"] = factor
    except Exception:
        pass  # keep 1.0 on any failure

    return result


def fetch_starters(game_date: date) -> list:
    """
    Fetch probable/confirmed starters + enrichment for all games on game_date.

    v3.0 additions per game:
      - confirmed_away / confirmed_home  (bool: starter status == "Confirmed")
      - away_pitcher_id / home_pitcher_id (int: for rest/lineup lookups)
      - ump_factor      (float: zone tendency multiplier)
      - away_rest / home_rest  (dict: days_rest, il_return)
      - away_lineup_factor / home_lineup_factor (float: platoon adj)
      - ump_name        (str: home plate ump name)
    """
    url = (f"{MLB_API}/schedule?sportId=1"
           f"&date={game_date.strftime('%Y-%m-%d')}"
           f"&hydrate=probablePitcher,teams,venue,gameInfo&gameType=R")
    data = _mlb_get(url)

    # Fetch umpire factors for the whole slate in one call
    ump_factors = fetch_umpire_factors(game_date)

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            aw = g["teams"]["away"]; hm = g["teams"]["home"]
            pk = g.get("gamePk")

            def code(t):
                ab = t["team"].get("abbreviation", "")
                nm = t["team"].get("name", "")
                return (MLB_TO_TRACKER.get(ab)
                        or MLB_NAME_TO_TRACKER.get(nm)
                        or ab or "UNK")

            gt = g.get("gameDate", "")
            try:
                game_time = datetime.fromisoformat(
                    gt.replace("Z", "+00:00")).strftime("%-I:%M %p")
            except Exception:
                game_time = "TBD"

            ap = aw.get("probablePitcher", {}); hp = hm.get("probablePitcher", {})

            away_name_str = ap.get("fullName", "TBD")
            home_name_str = hp.get("fullName", "TBD")
            away_pid = ap.get("id")
            home_pid = hp.get("id")

            # ── confirmed vs probable ───────────────────────────────────────
            # The schedule feed sets "status" on the probablePitcher object
            # when the starter has been officially confirmed.
            away_confirmed = ap.get("status", "").lower() == "confirmed"
            home_confirmed = hp.get("status", "").lower() == "confirmed"

            # ── days rest + IL return ───────────────────────────────────────
            away_rest = fetch_pitcher_rest(game_date, away_pid)
            home_rest = fetch_pitcher_rest(game_date, home_pid)

            # ── umpire factor ───────────────────────────────────────────────
            ump_factor = ump_factors.get(pk, 1.0)
            ump_name   = ump_factors.get(f"__name__{pk}", "")

            games.append({
                "game_time":         game_time,
                "game_pk":           pk,
                "away_team":         code(aw), "home_team":    code(hm),
                "away_name":         aw["team"].get("name", code(aw)),
                "home_name":         hm["team"].get("name", code(hm)),
                "away_pitcher":      away_name_str,
                "home_pitcher":      home_name_str,
                "away_pitcher_id":   away_pid,
                "home_pitcher_id":   home_pid,
                "away_confirmed":    away_confirmed,
                "home_confirmed":    home_confirmed,
                "away_rest":         away_rest,
                "home_rest":         home_rest,
                "ump_factor":        ump_factor,
                "ump_name":          ump_name,
                "venue":             g.get("venue", {}).get("name", ""),
            })

    # ── lineup handedness — done after all games collected ─────────────────
    # We need pitcher hand, which requires the xRA dict, so we defer this
    # to build_slate() where pitcher hand is already resolved.
    # Lineup data is fetched there via a helper.

    return games

# ── STEP 4: BUILD SLATE (v3.0) ────────────────────────────────────────────
def build_slate(games, pitcher_xra, bat_R, bat_L, bat_flat, bat_prior,
                league_bat, park_factors) -> pd.DataFrame:
    def _xra(name):
        return (pitcher_xra.get(name)
                or pitcher_xra.get(norm_name(name))
                or pitcher_xra.get(name.split()[-1])
                or LEAGUE_AVG)
    def _hand(name):
        return (pitcher_xra.get(f"__hand__{name}")
                or pitcher_xra.get(f"__hand__{norm_name(name)}", "R"))
    def _found(name):
        return name in pitcher_xra or norm_name(name) in pitcher_xra
    def _bat(team, hand, is_home):
        src = bat_L if hand == "L" else bat_R
        return src.get(team, bat_flat.get(team, bat_prior.get(team, league_bat)))

    def rest_penalty(rest_info: dict) -> float:
        """
        Returns a multiplier > 1.0 to inflate pitcher xRA on bad rest.
        Short rest or IL return = pitcher more likely to allow runs.
        """
        factor = 1.0
        if rest_info.get("days_rest", 99) <= SHORT_REST_DAYS:
            factor = max(factor, SHORT_REST_PENALTY)
        if rest_info.get("il_return", False):
            factor = max(factor, IL_RETURN_PENALTY)
        return factor

    rows = []
    for g in games:
        away, home       = g["away_team"], g["home_team"]
        apxra_base       = _xra(g["away_pitcher"])
        hpxra_base       = _xra(g["home_pitcher"])
        away_hand        = _hand(g["away_pitcher"])
        home_hand        = _hand(g["home_pitcher"])

        # ── Feature 4: days-rest adjustment on pitcher xRA ─────────────────
        away_rest_mult = rest_penalty(g.get("away_rest", {}))
        home_rest_mult = rest_penalty(g.get("home_rest", {}))
        apxra          = apxra_base * away_rest_mult
        hpxra          = hpxra_base * home_rest_mult

        # ── Feature 5: lineup handedness (actual L/R vs pitcher hand) ──────
        lineup = fetch_lineup_handedness(g.get("game_pk"), away_hand, home_hand)
        away_lineup_factor = lineup["away_lineup_factor"]
        home_lineup_factor = lineup["home_lineup_factor"]

        # ── Batting context (hand-split averages) ──────────────────────────
        away_bat_base = _bat(away, home_hand, False)
        home_bat_base = _bat(home, away_hand, True)
        away_bat      = away_bat_base * away_lineup_factor
        home_bat      = home_bat_base * home_lineup_factor

        # ── Feature 3: umpire zone multiplier ──────────────────────────────
        ump_factor = g.get("ump_factor", 1.0)

        pf = park_factors.get(home, 1.0)

        # xRUNS formula — ump_factor applies symmetrically to both sides
        ax = hpxra * (away_bat / LEAGUE_AVG) ** ALPHA * F5_SCALE * pf * ump_factor
        hx = apxra * (home_bat / LEAGUE_AVG) ** ALPHA * F5_SCALE * pf * ump_factor
        ax = max(ax, 0.1); hx = max(hx, 0.1)

        aw = float(winpct_vec(np.array([ax]), np.array([hx]))[0])
        hw = 1.0 - aw

        # ── Feature 2: confirmed-starter flag ──────────────────────────────
        away_confirmed = g.get("away_confirmed", False)
        home_confirmed = g.get("home_confirmed", False)

        # Rest flags for display
        away_dr  = g.get("away_rest", {}).get("days_rest", 99)
        home_dr  = g.get("home_rest", {}).get("days_rest", 99)
        away_il  = g.get("away_rest", {}).get("il_return", False)
        home_il  = g.get("home_rest", {}).get("il_return", False)

        rows.append({
            "Game Time":          g["game_time"],
            "Away Team":          g["away_name"],   "Away Code":       away,
            "Away Pitcher":       g["away_pitcher"],"Away P Found":    _found(g["away_pitcher"]),
            "Away Confirmed":     away_confirmed,
            "Away pxRA":          round(apxra_base, 3),
            "Away Rest Mult":     round(away_rest_mult, 3),
            "Away Days Rest":     away_dr if away_dr < 99 else "—",
            "Away IL Return":     away_il,
            "Away Bat xR":        round(away_bat, 3),
            "Away Lineup Factor": round(away_lineup_factor, 3),
            "Away xRUNS":         round(ax, 3),
            "Away Win%":          round(aw * 100, 1),
            "Away FV":            prob_to_american(aw),
            "Home Team":          g["home_name"],   "Home Code":       home,
            "Home Pitcher":       g["home_pitcher"],"Home P Found":    _found(g["home_pitcher"]),
            "Home Confirmed":     home_confirmed,
            "Home pxRA":          round(hpxra_base, 3),
            "Home Rest Mult":     round(home_rest_mult, 3),
            "Home Days Rest":     home_dr if home_dr < 99 else "—",
            "Home IL Return":     home_il,
            "Home Bat xR":        round(home_bat, 3),
            "Home Lineup Factor": round(home_lineup_factor, 3),
            "Home xRUNS":         round(hx, 3),
            "Home Win%":          round(hw * 100, 1),
            "Home FV":            prob_to_american(hw),
            "Ump Factor":         round(ump_factor, 3),
            "Ump Name":           g.get("ump_name", ""),
            "Park Factor":        round(pf, 4),
            "Venue":              g["venue"],
        })
    return pd.DataFrame(rows)

# ── STEP 5: EXCEL OUTPUT ───────────────────────────────────────────────────
def write_excel(df: pd.DataFrame, game_date: date, path: Path) -> None:
    wb = Workbook(); ws = wb.active; ws.title = "F5 Slate"
    NAVY="1F3864"; BLUE="2E75B6"; AWAY_CLR="D6E4F0"; HOME_CLR="D5F5E3"
    WARN="FFF3CD"; WHITE="FFFFFF"; BLACK="000000"; GREY="F8F9FA"
    thin   = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    fill   = lambda c: PatternFill("solid", start_color=c, fgColor=c)

    ws.merge_cells("A1:Y1")
    ws["A1"].value     = f"ClutchSheets — F5 Innings Slate  |  {game_date.strftime('%B %d, %Y')}"
    ws["A1"].font      = Font(name="Arial", bold=True, size=14, color=WHITE)
    ws["A1"].fill      = fill(BLUE)
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:Y2")
    ws["A2"].value     = (f"Model v3.0: pitcher xRA(window={XRA_WINDOW}) × (opp_bat / {LEAGUE_AVG})^{ALPHA} × lineup × {F5_SCALE}"
                          f"  |  NegBin r={NEGBIN_R}  |  decay={DECAY}"
                          f"  |  ump/rest/lineup enrichment active")
    ws["A2"].font      = Font(name="Arial", italic=True, size=9, color="595959")
    ws["A2"].alignment = Alignment(horizontal="left")
    ws.row_dimensions[2].height = 14

    COLS = [
        ("Game Time",18),("Away Team",18),("Away Pitcher",20),("Away pxRA",9),
        ("Away Bat xR",9),("Away xRUNS",9),("Away Win%",8),("Away FV",8),
        ("vs",3),
        ("Home Team",18),("Home Pitcher",20),("Home pxRA",9),
        ("Home Bat xR",9),("Home xRUNS",9),("Home Win%",8),("Home FV",8),
        ("Ump Factor",8),("Park Factor",9),("Venue",22),
        ("Away Days Rest",10),("Away IL Return",10),("Away Confirmed",11),
        ("Home Days Rest",10),("Home IL Return",10),("Home Confirmed",11),
        ("Ump Name",22),
    ]
    for ci,(label,width) in enumerate(COLS,1):
        c = ws.cell(row=3, column=ci, value=label)
        c.font=Font(name="Arial",bold=True,size=10,color=WHITE)
        c.fill=fill(NAVY); c.alignment=Alignment(horizontal="center",vertical="center")
        c.border=border
        ws.column_dimensions[get_column_letter(ci)].width=width
    ws.row_dimensions[3].height=18

    for ri,row in df.iterrows():
        er = 4 + ri
        af = fill(AWAY_CLR if row["Away P Found"] else WARN)
        hf = fill(HOME_CLR if row["Home P Found"] else WARN)
        # Flag unconfirmed starters with a different shade
        if not row.get("Away Confirmed", True): af = fill("FFF0B3")
        if not row.get("Home Confirmed", True): hf = fill("FFF0B3")
        vals = [
            row["Game Time"],row["Away Team"],row["Away Pitcher"],
            row["Away pxRA"],row["Away Bat xR"],row["Away xRUNS"],
            f"{row['Away Win%']}%",
            f"+{row['Away FV']}" if row["Away FV"]>0 else str(row["Away FV"]),
            "vs",
            row["Home Team"],row["Home Pitcher"],row["Home pxRA"],
            row["Home Bat xR"],row["Home xRUNS"],f"{row['Home Win%']}%",
            f"+{row['Home FV']}" if row["Home FV"]>0 else str(row["Home FV"]),
            row["Ump Factor"],row["Park Factor"],row["Venue"],
            row.get("Away Days Rest","—"), "Y" if row.get("Away IL Return") else "",
            "✓" if row.get("Away Confirmed") else "P",
            row.get("Home Days Rest","—"), "Y" if row.get("Home IL Return") else "",
            "✓" if row.get("Home Confirmed") else "P",
            row.get("Ump Name",""),
        ]
        for ci,val in enumerate(vals,1):
            c=ws.cell(row=er,column=ci,value=val)
            c.font=Font(name="Arial",size=10); c.border=border
            c.alignment=Alignment(horizontal="center")
            if ci<=8:
                c.fill=af
                if ci==8: c.font=Font(name="Arial",size=10,bold=True,
                    color="1A5276" if row["Away Win%"]>=50 else BLACK)
            elif ci==9:
                c.fill=fill(GREY); c.font=Font(name="Arial",size=9,color="888888")
            elif ci<=16:
                c.fill=hf
                if ci==16: c.font=Font(name="Arial",size=10,bold=True,
                    color="1A5276" if row["Home Win%"]>=50 else BLACK)
            else:
                c.fill=fill(GREY)
                if ci==17:
                    if row["Park Factor"]>1.05:   c.font=Font(name="Arial",size=10,bold=True,color="C0392B")
                    elif row["Park Factor"]<0.95: c.font=Font(name="Arial",size=10,bold=True,color="1A5276")
        ws.row_dimensions[er].height=16

    lr = 4 + len(df) + 1
    ws.cell(row=lr,   column=1, value="★ Amber = pitcher not in xRA db (league avg used)  |  Yellow = starter not yet confirmed (Probable)").font=Font(name="Arial",size=9,italic=True,color="7F6000")
    ws.cell(row=lr+1, column=1, value="★ Park >1.05 = hitter-friendly (red)  |  <0.95 = pitcher-friendly (blue)  |  Ump >1 = expand zone  |  IL/Short-rest flags in cols T–Y").font=Font(name="Arial",size=9,italic=True,color="595959")
    ws.freeze_panes=ws.cell(row=4,column=1)
    wb.save(path)

# ── GIT AUTO-PUSH ─────────────────────────────────────────────────────────
GIT_REPO = Path(r"D:\Desktop\F5 Baseball Forecaster")

def git_push(message: str, files: list) -> None:
    """Stage specific files, commit, and push to GitHub."""
    try:
        # Stage only the specified files
        for f in files:
            subprocess.run(["git", "-C", str(GIT_REPO), "add", str(f)],
                           check=True, capture_output=True)
        # Check if there's anything to commit
        result = subprocess.run(
            ["git", "-C", str(GIT_REPO), "diff", "--cached", "--quiet"],
            capture_output=True)
        if result.returncode == 0:
            print("  Git: nothing new to commit")
            return
        subprocess.run(["git", "-C", str(GIT_REPO), "commit", "-m", message],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(GIT_REPO), "push"],
                       check=True, capture_output=True)
        print(f"  Git: pushed — {message}")
    except subprocess.CalledProcessError as e:
        print(f"  Git WARNING: push failed — {e}")

# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ClutchSheets F5 Pipeline")
    parser.add_argument("--date",     default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel output")
    args = parser.parse_args()

    game_date = (date.today() if not args.date
                 else datetime.strptime(args.date, "%Y-%m-%d").date())
    date_str  = game_date.strftime("%Y-%m-%d")

    print(f"\nClutchSheets F5 Pipeline  |  {game_date}")
    print("=" * 50)

    # Step 1 — rebuild xRA
    print("[1/4] Rebuilding pitcher xRA (decay={})...".format(DECAY))
    rebuild_pitcher_xra()

    # Step 2 — load data
    print("[2/4] Loading data...")
    pitcher_xra                          = load_pitcher_xra()
    bat_R, bat_L, bat_flat, bat_prior, league_bat = load_batting_xruns()
    park_factors                         = load_park_factors()
    print(f"  Pitchers: {len([k for k in pitcher_xra if not k.startswith('__')])}  "
          f"bat_R:{len(bat_R)}  bat_L:{len(bat_L)}  parks:{len(park_factors)}")

    # Step 3 — fetch starters
    print(f"[3/4] Fetching starters + enrichment for {game_date}...")
    games = fetch_starters(game_date)
    print(f"  Games: {len(games)}")
    if not games:
        print("  No games found. Exiting."); sys.exit(0)

    all_p   = [g["away_pitcher"] for g in games] + [g["home_pitcher"] for g in games]
    missing = [p for p in all_p if p != "TBD"
               and p not in pitcher_xra and norm_name(p) not in pitcher_xra]
    print(f"  Pitchers found: {len(all_p)-len(missing)}/{len(all_p)}")
    if missing: print(f"  Missing (league avg): {', '.join(missing)}")

    # Confirmed-starter summary
    unconfirmed = [
        f"{g['away_pitcher']}({g['away_team']})" for g in games if not g.get("away_confirmed")
    ] + [
        f"{g['home_pitcher']}({g['home_team']})" for g in games if not g.get("home_confirmed")
    ]
    if unconfirmed:
        print(f"  ⚠ Probable (not confirmed): {', '.join(unconfirmed)}")
    else:
        print("  All starters confirmed")

    # Step 4 — compute + output
    print("[4/4] Computing projections and saving outputs...")
    df = build_slate(games, pitcher_xra, bat_R, bat_L, bat_flat,
                     bat_prior, league_bat, park_factors)

    print("\n" + "=" * 68)
    print(f"{'GAME':<36} {'AWAY FV':>8} {'AWAY W%':>8} {'HOME FV':>8} {'HOME W%':>8}")
    print("=" * 68)
    for _, row in df.iterrows():
        gs  = f"{row['Away Team']} @ {row['Home Team']}"
        afv = f"+{row['Away FV']}" if row["Away FV"]>0 else str(row["Away FV"])
        hfv = f"+{row['Home FV']}" if row["Home FV"]>0 else str(row["Home FV"])
        print(f"{gs:<36} {afv:>8} {row['Away Win%']:>7}% {hfv:>8} {row['Home Win%']:>7}%")
    print("=" * 68)

    # Detailed CSV
    detail_cols = [
        "Game Time","Away Team","Away Pitcher","Away Confirmed",
        "Away pxRA","Away Rest Mult","Away Days Rest","Away IL Return",
        "Away Bat xR","Away Lineup Factor","Away xRUNS","Away Win%","Away FV",
        "Home Team","Home Pitcher","Home Confirmed",
        "Home pxRA","Home Rest Mult","Home Days Rest","Home IL Return",
        "Home Bat xR","Home Lineup Factor","Home xRUNS","Home Win%","Home FV",
        "Ump Factor","Ump Name","Park Factor","Venue",
    ]
    df[detail_cols].to_csv(MODEL_DIR / f"f5_slate_{date_str}.csv", index=False)
    print(f"\nSlate CSV:    {MODEL_DIR / f'f5_slate_{date_str}.csv'}")

    # Pipeline CSV
    pd.DataFrame({
        "date":         date_str,
        "away_team":    df["Away Code"].values,
        "away_sp":      df["Away Pitcher"].values,
        "home_team":    df["Home Code"].values,
        "home_sp":      df["Home Pitcher"].values,
        "lambda_away":  df["Away xRUNS"].round(3).values,
        "lambda_home":  df["Home xRUNS"].round(3).values,
        "total_proj":   (df["Away xRUNS"]+df["Home xRUNS"]).round(3).values,
        "winprob_home": (df["Home Win%"]/100).round(3).values,
        "model_version":"v3.0",
    }).to_csv(FORECAST_DIR / f"f5_forecast_{date_str}.csv", index=False)
    print(f"Forecast CSV: {FORECAST_DIR / f'f5_forecast_{date_str}.csv'}")

    # Excel
    if not args.no_excel:
        xlsx = MODEL_DIR / f"f5_slate_{date_str}.xlsx"
        try:
            write_excel(df, game_date, xlsx)
            print(f"Excel:        {xlsx}")
        except PermissionError:
            print(f"Excel:        SKIPPED (file open in Excel — close it to regenerate)")

    # Write daily_forecast.csv to repo root (this is what app.py + GitHub expect)
    daily_csv = FORECAST_DIR / "daily_forecast.csv"
    pd.DataFrame({
        "date":         date_str,
        "away_team":    df["Away Code"].values,
        "away_sp":      df["Away Pitcher"].values,
        "home_team":    df["Home Code"].values,
        "home_sp":      df["Home Pitcher"].values,
        "lambda_away":  df["Away xRUNS"].round(3).values,
        "lambda_home":  df["Home xRUNS"].round(3).values,
        "total_proj":   (df["Away xRUNS"]+df["Home xRUNS"]).round(3).values,
        "winprob_home": (df["Home Win%"]/100).round(3).values,
        "model_version":"v3.0",
    }).to_csv(daily_csv, index=False)
    print(f"daily_forecast.csv: {daily_csv}")

    print("\nPushing to GitHub...")
    git_push(f"forecast {date_str}", [daily_csv])

    print("\nDone.\n")

if __name__ == "__main__":
    main()
