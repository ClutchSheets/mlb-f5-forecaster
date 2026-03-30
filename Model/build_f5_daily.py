"""
build_f5_daily.py  —  ClutchSheets F5 Daily Workbook
=====================================================================
Reads today's pipeline slate CSV and builds a fully self-contained
Excel workbook with all lookup data embedded.

Sheets:
  Slate        — up to 15 games pre-filled, XLOOKUP-driven
  Pitchers     — name / throws / f5_ra / starts  (XLOOKUP source)
  TeamOffense  — team / vs_RHP / vs_LHP          (XLOOKUP source)
  ParkFactors  — team / park_factor              (XLOOKUP source)
  Umpires      — ump_name / ump_run_delta        (XLOOKUP source)
  Forecast     — daily_forecast.csv layout, formula-driven from Slate

Daily workflow:
  python f5_pipeline.py        ← fetches starters, builds f5_slate_YYYY-MM-DD.csv
  python build_f5_daily.py     ← pre-fills workbook from slate
  [open workbook, swap pitchers, adjust umps, override xRUNS]
  python push_from_slate.py    ← Forecast sheet → daily_forecast.csv → git push

Usage:
  python build_f5_daily.py
  python build_f5_daily.py --date 2026-04-01
"""

import argparse
import csv
import unicodedata
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ── PATHS ──────────────────────────────────────────────────────────────────
MODEL_DIR    = Path(r"D:\Desktop\F5 Baseball Forecaster\Model")
FORECAST_DIR = Path(r"D:\Desktop\F5 Baseball Forecaster")

# ── MODEL CONSTANTS ────────────────────────────────────────────────────────
LEAGUE_AVG = 2.52
ALPHA      = 1.2
F5_SCALE   = 0.80
NEGBIN_R   = 6.0

# ── COLOURS ────────────────────────────────────────────────────────────────
NAVY   = "1F3864"
BLUE   = "2E75B6"
LTBLU  = "D6E4F0"
GREEN  = "D5F5E3"
AMBER  = "FFF3CD"
GREY   = "F2F2F2"
WHITE  = "FFFFFF"
BLACK  = "000000"
RED    = "C0392B"
DKBLU  = "1A5276"
LTGRN  = "A9DFBF"

# ── STYLE HELPERS ──────────────────────────────────────────────────────────
def fill(h):
    return PatternFill("solid", start_color=h, fgColor=h)

def font(bold=False, color=BLACK, size=10, italic=False):
    return Font(name="Arial", bold=bold, color=color, size=size, italic=italic)

def border():
    t = Side(style="thin", color="BFBFBF")
    return Border(left=t, right=t, top=t, bottom=t)

def ctr():  return Alignment(horizontal="center", vertical="center")
def lft():  return Alignment(horizontal="left",   vertical="center")

BD = border()


def hdr_cell(ws, row, col, label, bg=NAVY, fg=WHITE, width=None, sz=10, bold=True):
    c = ws.cell(row=row, column=col, value=label)
    c.font = font(bold=bold, color=fg, size=sz)
    c.fill = fill(bg)
    c.alignment = ctr()
    c.border = BD
    if width:
        ws.column_dimensions[get_column_letter(col)].width = width
    return c


def data_cell(ws, row, col, value, bg=GREY, bold=False, fmt=None, align="center", color=BLACK):
    c = ws.cell(row=row, column=col, value=value)
    c.font = font(bold=bold, color=color)
    c.fill = fill(bg)
    c.alignment = ctr() if align == "center" else lft()
    c.border = BD
    if fmt:
        c.number_format = fmt
    return c


def formula_cell(ws, row, col, formula, bg=LTBLU, fmt=None, bold=False, color=BLACK):
    c = ws.cell(row=row, column=col, value=formula)
    c.font = font(bold=bold, color=color)
    c.fill = fill(bg)
    c.alignment = ctr()
    c.border = BD
    if fmt:
        c.number_format = fmt
    return c


def norm(s):
    """Strip accents for fuzzy name matching."""
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn")


# ── LOAD SOURCE DATA ───────────────────────────────────────────────────────
def load_pitchers():
    path = MODEL_DIR / "f5_ra_by_pitcher.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["name"]   = df["name"].str.strip()
    df["f5_ra"]  = pd.to_numeric(df["f5_ra"],  errors="coerce").round(4)
    df["starts"] = pd.to_numeric(df["starts"], errors="coerce").fillna(0).astype(int)

    # Throws from tracker
    tr_path = MODEL_DIR / "f5_ra_tracker.csv"
    hand = {}
    if tr_path.exists():
        tr = pd.read_csv(tr_path)
        tr.columns  = tr.columns.str.strip()
        tr["pitcher"] = tr["pitcher"].str.strip()
        tr["throws"]  = tr["throws"].str.strip().str.upper()
        hand = tr.groupby("pitcher")["throws"].agg(
            lambda x: x.mode()[0] if len(x) else "R").to_dict()

    df["throws"] = df["name"].map(hand).fillna("R")
    return df.sort_values("name").reset_index(drop=True)


def load_team_offense():
    path = MODEL_DIR / "f5_rs_tracker.csv"
    rs = pd.read_csv(path)
    rs.columns   = rs.columns.str.strip()
    rs["date"]     = pd.to_datetime(rs["date"])
    rs["season"]   = rs["date"].dt.year
    rs["team"]     = rs["team"].str.strip().str.upper()
    rs["opp_hand"] = rs["opp_hand"].str.strip().str.upper()

    counts     = rs.groupby("season").size()
    cur_season = counts[counts >= 200].index.max()
    cur        = rs[rs["season"] == cur_season]

    split    = cur.groupby(["team", "opp_hand"])["f5_scored"].mean().reset_index()
    bat_R    = split[split["opp_hand"] == "R"].set_index("team")["f5_scored"].to_dict()
    bat_L    = split[split["opp_hand"] == "L"].set_index("team")["f5_scored"].to_dict()
    bat_flat = cur.groupby("team")["f5_scored"].mean().to_dict()

    teams = sorted(set(list(bat_R) + list(bat_L) + list(bat_flat)))
    rows = []
    for t in teams:
        rows.append({
            "Team":   t,
            "vs_RHP": round(bat_R.get(t, bat_flat.get(t, LEAGUE_AVG)), 4),
            "vs_LHP": round(bat_L.get(t, bat_flat.get(t, LEAGUE_AVG)), 4),
        })
    return pd.DataFrame(rows)


def load_park_factors():
    path = MODEL_DIR / "f5_park_factors.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Team", "park_factor"])
    df = pd.read_csv(path)
    df.columns  = df.columns.str.strip()
    df["Team"]  = df["Team"].str.strip()
    return df[["Team", "park_factor"]].sort_values("Team").reset_index(drop=True)


def load_umpires():
    path = MODEL_DIR / "f5_ump_ratings.csv"
    if not path.exists():
        return pd.DataFrame(columns=["umpire_name", "ump_run_delta"])
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["umpire_name"]   = df["umpire_name"].str.strip()
    df["ump_run_delta"] = pd.to_numeric(df["ump_run_delta"], errors="coerce").round(4)
    return df[["umpire_name", "ump_run_delta"]].sort_values("umpire_name").reset_index(drop=True)


def load_slate(date_str):
    path = MODEL_DIR / f"f5_slate_{date_str}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── SHEET: Pitchers ────────────────────────────────────────────────────────
def write_pitchers(wb, df):
    ws = wb.create_sheet("Pitchers")

    # Title
    ws.merge_cells("A1:D1")
    ws["A1"].value     = "Pitcher xRA Database  —  update via f5_pipeline.py"
    ws["A1"].font      = font(bold=True, color=WHITE, size=11)
    ws["A1"].fill      = fill(NAVY)
    ws["A1"].alignment = lft()
    ws.row_dimensions[1].height = 20

    # Headers — col A=Name, B=Throws, C=f5_ra, D=Starts
    for ci, (label, width) in enumerate(
            [("Name",24),("Throws",8),("f5_ra",9),("Starts",8)], 1):
        hdr_cell(ws, 2, ci, label, width=width)
    ws.row_dimensions[2].height = 16

    for i, row in df.iterrows():
        r = 3 + i
        c = ws.cell(row=r, column=1, value=str(row["name"]))
        c.font = font(size=10); c.fill = fill(WHITE)
        c.alignment = lft(); c.border = BD
        data_cell(ws, r, 2, str(row["throws"]),    bg=GREY)
        data_cell(ws, r, 3, float(row["f5_ra"]),   bg=LTBLU, fmt="0.000")
        data_cell(ws, r, 4, int(row["starts"]),    bg=GREY)
        ws.row_dimensions[r].height = 15

    ws.freeze_panes = ws["A3"]
    print(f"  Pitchers sheet: {len(df)} rows")
    return 3 + len(df) - 1   # last data row


# ── SHEET: TeamOffense ─────────────────────────────────────────────────────
def write_team_offense(wb, df):
    ws = wb.create_sheet("TeamOffense")

    ws.merge_cells("A1:C1")
    ws["A1"].value     = "Team F5 Batting Splits  —  update via f5_pipeline.py"
    ws["A1"].font      = font(bold=True, color=WHITE, size=11)
    ws["A1"].fill      = fill(NAVY)
    ws["A1"].alignment = lft()
    ws.row_dimensions[1].height = 20

    for ci, (label, width) in enumerate(
            [("Team",9),("vs_RHP",11),("vs_LHP",11)], 1):
        hdr_cell(ws, 2, ci, label, width=width)
    ws.row_dimensions[2].height = 16

    for i, row in df.iterrows():
        r = 3 + i
        c = ws.cell(row=r, column=1, value=str(row["Team"]))
        c.font = font(size=10); c.fill = fill(WHITE)
        c.alignment = ctr(); c.border = BD
        data_cell(ws, r, 2, float(row["vs_RHP"]), bg=LTBLU, fmt="0.000")
        data_cell(ws, r, 3, float(row["vs_LHP"]), bg=GREEN,  fmt="0.000")
        ws.row_dimensions[r].height = 15

    ws.freeze_panes = ws["A3"]
    print(f"  TeamOffense sheet: {len(df)} rows")
    return 3 + len(df) - 1


# ── SHEET: ParkFactors ─────────────────────────────────────────────────────
def write_park_factors(wb, df):
    ws = wb.create_sheet("ParkFactors")

    ws.merge_cells("A1:B1")
    ws["A1"].value     = "Park Factors  —  update via f5_pipeline.py"
    ws["A1"].font      = font(bold=True, color=WHITE, size=11)
    ws["A1"].fill      = fill(NAVY)
    ws["A1"].alignment = lft()
    ws.row_dimensions[1].height = 20

    hdr_cell(ws, 2, 1, "Team",        width=10)
    hdr_cell(ws, 2, 2, "Park Factor", width=13)
    ws.row_dimensions[2].height = 16

    for i, row in df.iterrows():
        r = 3 + i
        pf = float(row["park_factor"])
        c = ws.cell(row=r, column=1, value=str(row["Team"]))
        c.font = font(size=10); c.fill = fill(WHITE)
        c.alignment = ctr(); c.border = BD
        pc = ws.cell(row=r, column=2, value=round(pf, 4))
        pc.border = BD; pc.alignment = ctr(); pc.number_format = "0.000"
        pc.font = font(bold=True,
                       color=RED   if pf > 1.05 else
                             DKBLU if pf < 0.95 else BLACK)
        pc.fill = fill(GREY)
        ws.row_dimensions[r].height = 15

    ws.freeze_panes = ws["A3"]
    print(f"  ParkFactors sheet: {len(df)} rows")
    return 3 + len(df) - 1


# ── SHEET: Umpires ─────────────────────────────────────────────────────────
def write_umpires(wb, df):
    ws = wb.create_sheet("Umpires")

    ws.merge_cells("A1:C1")
    ws["A1"].value     = "Umpire Run Delta  —  update via build_ump_ratings.py"
    ws["A1"].font      = font(bold=True, color=WHITE, size=11)
    ws["A1"].fill      = fill(NAVY)
    ws["A1"].alignment = lft()
    ws.row_dimensions[1].height = 20

    ws.merge_cells("A2:C2")
    ws["A2"].value = (
        "ump_run_delta = avg_favor / 2  (per-team per-game runs vs average).  "
        ">+0.20 = expand zone  |  <−0.10 = squeeze zone"
    )
    ws["A2"].font = font(italic=True, size=9, color="595959")
    ws["A2"].alignment = lft()
    ws.row_dimensions[2].height = 13

    hdr_cell(ws, 3, 1, "Umpire Name",    width=26)
    hdr_cell(ws, 3, 2, "ump_run_delta",  width=15)
    hdr_cell(ws, 3, 3, "Zone",           width=10)
    ws.row_dimensions[3].height = 16

    if df.empty:
        ws.cell(row=4, column=1).value = "Run build_ump_ratings.py to populate"
        ws.cell(row=4, column=1).font  = font(italic=True, color="888888")
        print("  Umpires sheet: empty (run build_ump_ratings.py)")
        return 4

    for i, row in df.iterrows():
        r = 4 + i
        delta = float(row["ump_run_delta"])
        zone  = ("EXPAND" if delta >= 0.20 else
                 "squeeze" if delta <= -0.10 else "neutral")
        zcolor = (RED   if zone == "EXPAND" else
                  DKBLU if zone == "squeeze" else "595959")

        c = ws.cell(row=r, column=1, value=str(row["umpire_name"]))
        c.font = font(size=10); c.fill = fill(WHITE)
        c.alignment = lft(); c.border = BD

        dc = ws.cell(row=r, column=2, value=delta)
        dc.font = font(bold=True,
                       color=RED   if delta >= 0.20 else
                             DKBLU if delta <= -0.10 else BLACK)
        dc.fill = fill(GREY); dc.border = BD
        dc.alignment = ctr(); dc.number_format = "0.000"

        zc = ws.cell(row=r, column=3, value=zone)
        zc.font = font(bold=(zone != "neutral"), color=zcolor, size=9)
        zc.fill = fill(GREY); zc.border = BD; zc.alignment = ctr()
        ws.row_dimensions[r].height = 15

    ws.freeze_panes = ws["A4"]
    print(f"  Umpires sheet: {len(df)} rows")
    return 4 + len(df) - 1


# ── SHEET: Slate ───────────────────────────────────────────────────────────
def write_slate(wb, games, date_str, last_rows):
    ws = wb.create_sheet("Slate", 0)

    p_last  = last_rows["pitchers"]
    t_last  = last_rows["teams"]
    pf_last = last_rows["parks"]
    u_last  = last_rows["umps"]

    game_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    MAX_GAMES = 15
    N_GAMES   = min(len(games), MAX_GAMES)

    # ── Row 1: Title ───────────────────────────────────────────────────────
    ws.merge_cells("A1:W1")
    ws["A1"].value = (
        f"ClutchSheets  |  F5 Innings  |  {game_date.strftime('%B %d, %Y')}  "
        f"|  xRA × (oppBat/{LEAGUE_AVG})^{ALPHA} × {F5_SCALE}  |  NegBin r={NEGBIN_R}"
    )
    ws["A1"].font      = font(bold=True, color=WHITE, size=12)
    ws["A1"].fill      = fill(BLUE)
    ws["A1"].alignment = ctr()
    ws.row_dimensions[1].height = 26

    # ── Row 2: Instructions ────────────────────────────────────────────────
    ws.merge_cells("A2:W2")
    ws["A2"].value = (
        "✏  YELLOW = type here (team codes, pitcher names, ump name).  "
        "BLUE = auto via XLOOKUP — updates when you change any yellow cell.  "
        "OVR cols override final xRUNS.  Win% uses pipeline values — re-run pipeline to refresh."
    )
    ws["A2"].font      = font(italic=True, size=9, color="595959")
    ws["A2"].alignment = lft()
    ws.row_dimensions[2].height = 14

    # ── Row 3: Column headers ──────────────────────────────────────────────
    # A=# B=Time C=Away(YELLOW) D=Away SP(YELLOW) E=Throws(auto)
    # F=Away xRA(auto) G=Away Bat(auto) H=Away xR(auto) I=OVR(YELLOW)
    # J=Away W% K=Away FV  L=vs
    # M=Home(YELLOW) N=Home SP(YELLOW) O=Throws(auto)
    # P=Home xRA(auto) Q=Home Bat(auto) R=Home xR(auto) S=OVR(YELLOW)
    # T=Home W% U=Home FV  V=Ump(YELLOW)  W=Ump Δ(auto)  X=Park(auto)

    COLS = [
        (1,  "#",         3,  NAVY,  WHITE),
        (2,  "Time",      9,  NAVY,  WHITE),
        (3,  "Away",      7,  AMBER, BLACK),
        (4,  "Away SP",   22, AMBER, BLACK),
        (5,  "Throws",    7,  LTBLU, BLACK),
        (6,  "Away xRA",  9,  LTBLU, BLACK),
        (7,  "Away Bat",  9,  LTBLU, BLACK),
        (8,  "Away xR",   9,  LTBLU, BLACK),
        (9,  "OVR ▼",     8,  AMBER, BLACK),
        (10, "Away W%",   9,  NAVY,  WHITE),
        (11, "Away FV",   8,  NAVY,  WHITE),
        (12, "vs",        3,  GREY,  "888888"),
        (13, "Home",      7,  AMBER, BLACK),
        (14, "Home SP",   22, AMBER, BLACK),
        (15, "Throws",    7,  LTBLU, BLACK),
        (16, "Home xRA",  9,  LTBLU, BLACK),
        (17, "Home Bat",  9,  LTBLU, BLACK),
        (18, "Home xR",   9,  LTBLU, BLACK),
        (19, "OVR ▼",     8,  AMBER, BLACK),
        (20, "Home W%",   9,  GREEN, BLACK),
        (21, "Home FV",   8,  GREEN, BLACK),
        (22, "Ump",       22, AMBER, BLACK),
        (23, "Ump Δ",     8,  LTBLU, BLACK),
        (24, "Park",      8,  LTBLU, BLACK),
    ]
    for col, label, width, bg, fg in COLS:
        hdr_cell(ws, 3, col, label, bg=bg, fg=fg, width=width)
    ws.row_dimensions[3].height = 18

    # ── XLOOKUP range references ───────────────────────────────────────────
    # Pitchers:    col A = name, col B = throws, col C = f5_ra
    P_NAMES  = f"Pitchers!$A$3:$A${p_last}"
    P_THROWS = f"Pitchers!$B$3:$B${p_last}"
    P_XRA    = f"Pitchers!$C$3:$C${p_last}"

    # TeamOffense: col A = team, col B = vs_RHP, col C = vs_LHP
    T_TEAMS  = f"TeamOffense!$A$3:$A${t_last}"
    T_RHP    = f"TeamOffense!$B$3:$B${t_last}"
    T_LHP    = f"TeamOffense!$C$3:$C${t_last}"

    # ParkFactors: col A = team, col B = park_factor
    PF_TEAMS = f"ParkFactors!$A$3:$A${pf_last}"
    PF_VALS  = f"ParkFactors!$B$3:$B${pf_last}"

    # Umpires: col A = name, col B = ump_run_delta
    U_NAMES  = f"Umpires!$A$4:$A${u_last}"
    U_DELTA  = f"Umpires!$B$4:$B${u_last}"

    # ── Data rows ──────────────────────────────────────────────────────────
    for gi in range(MAX_GAMES):
        r = 4 + gi
        g = games[gi] if gi < len(games) else {}

        away_team  = g.get("Away Team",  g.get("Away Code",  "")) or ""
        home_team  = g.get("Home Team",  g.get("Home Code",  "")) or ""
        away_sp    = g.get("Away Pitcher", "") or ""
        home_sp    = g.get("Home Pitcher", "") or ""
        game_time  = g.get("Game Time",    "") or ""
        ump_name   = g.get("Ump Name",     "") or ""
        aw_pct     = float(g.get("Away Win%", 50) or 50) / 100
        hw_pct     = float(g.get("Home Win%", 50) or 50) / 100

        # Away team code — pipeline gives full name in Away Team col
        # and code in Away Code col
        away_code = g.get("Away Code", away_team) or away_team
        home_code = g.get("Home Code", home_team) or home_team

        # XLOOKUP formulas
        # Throws: exact match, fallback to approximate (match_mode=2), default "R"
        away_throws_f = (f'=IFERROR(XLOOKUP(D{r},{P_NAMES},{P_THROWS},"R",0),'
                         f'IFERROR(XLOOKUP(D{r},{P_NAMES},{P_THROWS},"R",2),"R"))')
        home_throws_f = (f'=IFERROR(XLOOKUP(N{r},{P_NAMES},{P_THROWS},"R",0),'
                         f'IFERROR(XLOOKUP(N{r},{P_NAMES},{P_THROWS},"R",2),"R"))')

        # xRA: exact then approximate, default LEAGUE_AVG
        away_xra_f = (f'=IFERROR(XLOOKUP(D{r},{P_NAMES},{P_XRA},{LEAGUE_AVG},0),'
                      f'IFERROR(XLOOKUP(D{r},{P_NAMES},{P_XRA},{LEAGUE_AVG},2),{LEAGUE_AVG}))')
        home_xra_f = (f'=IFERROR(XLOOKUP(N{r},{P_NAMES},{P_XRA},{LEAGUE_AVG},0),'
                      f'IFERROR(XLOOKUP(N{r},{P_NAMES},{P_XRA},{LEAGUE_AVG},2),{LEAGUE_AVG}))')

        # Batting split: away bats vs home pitcher throws (col O)
        # IF home throws = L → use vs_LHP, else vs_RHP
        away_bat_f = (f'=IFERROR(IF(O{r}="L",'
                      f'XLOOKUP(C{r},{T_TEAMS},{T_LHP},{LEAGUE_AVG}),'
                      f'XLOOKUP(C{r},{T_TEAMS},{T_RHP},{LEAGUE_AVG})),'
                      f'{LEAGUE_AVG})')
        home_bat_f = (f'=IFERROR(IF(E{r}="L",'
                      f'XLOOKUP(M{r},{T_TEAMS},{T_LHP},{LEAGUE_AVG}),'
                      f'XLOOKUP(M{r},{T_TEAMS},{T_RHP},{LEAGUE_AVG})),'
                      f'{LEAGUE_AVG})')

        # Park factor — keyed on home team code
        pf_f = f'=IFERROR(XLOOKUP(M{r},{PF_TEAMS},{PF_VALS},1),1)'

        # Ump run delta — keyed on ump name in col V
        ump_delta_f = f'=IFERROR(XLOOKUP(V{r},{U_NAMES},{U_DELTA},0),0)'

        # xRUNS = pitcher_xRA * (opp_bat/LEAGUE_AVG)^ALPHA * F5_SCALE * park_factor
        # Auto xRUNS uses the park factor cell
        away_xr_f = (f'=IFERROR(ROUND(F{r}*(G{r}/{LEAGUE_AVG})^{ALPHA}'
                     f'*{F5_SCALE}*X{r},3),"")')
        home_xr_f = (f'=IFERROR(ROUND(P{r}*(Q{r}/{LEAGUE_AVG})^{ALPHA}'
                     f'*{F5_SCALE}*X{r},3),"")')

        # Final xRUNS = OVR if filled, else auto
        away_final = f'=IF(I{r}<>"",I{r},H{r})'
        home_final = f'=IF(S{r}<>"",S{r},R{r})'

        # Win% — pipeline values stored as numbers; when OVR touched, show note
        # We store pipeline win% directly; user re-runs pipeline to update
        away_wp = round(aw_pct, 4) if g else ""
        home_wp = round(hw_pct, 4) if g else ""

        # FV (American odds)
        away_fv_f = (f'=IFERROR(IF(J{r}>=0.5,'
                     f'-ROUND(J{r}/(1-J{r})*100,0),'
                     f'ROUND((1-J{r})/J{r}*100,0)),"")')
        home_fv_f = (f'=IFERROR(IF(T{r}>=0.5,'
                     f'-ROUND(T{r}/(1-T{r})*100,0),'
                     f'ROUND((1-T{r})/T{r}*100,0)),"")')

        # ── Write cells ────────────────────────────────────────────────────
        row_bg = WHITE if gi % 2 == 0 else GREY

        data_cell(ws, r, 1,  gi + 1,      bg=GREY)
        data_cell(ws, r, 2,  game_time,   bg=GREY)
        data_cell(ws, r, 3,  away_code,   bg=AMBER, align="center")    # Away team
        data_cell(ws, r, 4,  away_sp,     bg=AMBER, align="left")      # Away SP ← edit
        formula_cell(ws, r, 5,  away_throws_f, bg=LTBLU)               # Throws auto
        formula_cell(ws, r, 6,  away_xra_f,    bg=LTBLU, fmt="0.000")  # xRA auto
        formula_cell(ws, r, 7,  away_bat_f,    bg=LTBLU, fmt="0.000")  # Bat auto
        formula_cell(ws, r, 8,  away_xr_f,     bg=LTBLU, fmt="0.000")  # xRUNS auto
        data_cell(ws, r, 9,  "",          bg=AMBER, fmt="0.000")        # OVR ← edit
        # Win% — store pipeline value; highlight amber if OVR touched
        wp_cell_a = ws.cell(row=r, column=10, value=away_wp)
        wp_cell_a.font = font(bold=True, color=WHITE)
        wp_cell_a.fill = fill(NAVY); wp_cell_a.border = BD
        wp_cell_a.alignment = ctr(); wp_cell_a.number_format = "0.0%"
        formula_cell(ws, r, 11, away_fv_f, bg=NAVY,
                     bold=True, color=WHITE)                            # Away FV

        vs_c = ws.cell(row=r, column=12, value="vs")
        vs_c.font = font(color="888888", size=9)
        vs_c.fill = fill(GREY); vs_c.border = BD; vs_c.alignment = ctr()

        data_cell(ws, r, 13, home_code,   bg=AMBER, align="center")    # Home team
        data_cell(ws, r, 14, home_sp,     bg=AMBER, align="left")      # Home SP ← edit
        formula_cell(ws, r, 15, home_throws_f, bg=LTBLU)               # Throws auto
        formula_cell(ws, r, 16, home_xra_f,    bg=LTBLU, fmt="0.000")  # xRA auto
        formula_cell(ws, r, 17, home_bat_f,    bg=LTBLU, fmt="0.000")  # Bat auto
        formula_cell(ws, r, 18, home_xr_f,     bg=LTBLU, fmt="0.000")  # xRUNS auto
        data_cell(ws, r, 19, "",          bg=AMBER, fmt="0.000")        # OVR ← edit
        wp_cell_h = ws.cell(row=r, column=20, value=home_wp)
        wp_cell_h.font = font(bold=True, color=BLACK)
        wp_cell_h.fill = fill(GREEN); wp_cell_h.border = BD
        wp_cell_h.alignment = ctr(); wp_cell_h.number_format = "0.0%"
        formula_cell(ws, r, 21, home_fv_f, bg=GREEN, bold=True)        # Home FV

        data_cell(ws, r, 22, ump_name,    bg=AMBER, align="left")      # Ump ← edit
        formula_cell(ws, r, 23, ump_delta_f, bg=LTBLU, fmt="0.000")    # Ump Δ auto
        formula_cell(ws, r, 24, pf_f,     bg=LTBLU, fmt="0.000")       # Park auto

        ws.row_dimensions[r].height = 17

    # ── Legend ─────────────────────────────────────────────────────────────
    lr = 4 + MAX_GAMES + 1
    ws.merge_cells(f"A{lr}:X{lr}")
    ws[f"A{lr}"].value = (
        "★ YELLOW = type here  ★ BLUE = XLOOKUP auto  "
        "★ OVR (cols I/S) overrides final xRUNS in Forecast  "
        "★ Win% uses pipeline values — re-run f5_pipeline.py + build_f5_daily.py to refresh  "
        "★ Park red >1.05 = hitter-friendly  |  blue <0.95 = pitcher-friendly"
    )
    ws[f"A{lr}"].font = font(italic=True, size=9, color="595959")
    ws[f"A{lr}"].alignment = lft()
    ws.row_dimensions[lr].height = 13

    ws.freeze_panes = ws["A4"]
    print(f"  Slate sheet: {N_GAMES} games pre-filled ({MAX_GAMES - N_GAMES} empty rows ready)")


# ── SHEET: Forecast ────────────────────────────────────────────────────────
def write_forecast(wb, games, date_str):
    ws = wb.create_sheet("Forecast")

    MAX_GAMES = 15

    ws.merge_cells("A1:J1")
    ws["A1"].value = (
        f"daily_forecast.csv  |  {date_str}  |  "
        "Formula-driven from Slate.  Run push_from_slate.py to export."
    )
    ws["A1"].font = font(italic=True, size=9, color="595959")
    ws["A1"].alignment = lft(); ws.row_dimensions[1].height = 14

    HEADS = ["date","away_team","away_sp","home_team","home_sp",
             "lambda_away","lambda_home","total_proj","winprob_home","model_version"]
    WIDTHS = [12, 9, 22, 9, 22, 11, 11, 11, 13, 12]
    for ci, (h, w) in enumerate(zip(HEADS, WIDTHS), 1):
        hdr_cell(ws, 2, ci, h, width=w)
    ws.row_dimensions[2].height = 16

    for gi in range(MAX_GAMES):
        sr = 4 + gi   # Slate row
        fr = 3 + gi   # Forecast row

        # Use OVR if filled, else auto xRUNS
        lam_a = f"=IF(Slate!I{sr}<>\"\",ROUND(Slate!I{sr},3),ROUND(Slate!H{sr},3))"
        lam_h = f"=IF(Slate!S{sr}<>\"\",ROUND(Slate!S{sr},3),ROUND(Slate!R{sr},3))"

        vals = [
            date_str,
            f"=IF(Slate!C{sr}=\"\",\"\",Slate!C{sr})",
            f"=IF(Slate!D{sr}=\"\",\"\",Slate!D{sr})",
            f"=IF(Slate!M{sr}=\"\",\"\",Slate!M{sr})",
            f"=IF(Slate!N{sr}=\"\",\"\",Slate!N{sr})",
            lam_a,
            lam_h,
            f"=IF(Slate!M{sr}=\"\",\"\",ROUND({lam_a[1:]}+{lam_h[1:]},3))",
            f"=IF(Slate!T{sr}=\"\",0.5,ROUND(Slate!T{sr},4))",
            "v3.0",
        ]
        FMTS = [None,None,None,None,None,"0.000","0.000","0.000","0.000",None]
        for ci, (v, fmt) in enumerate(zip(vals, FMTS), 1):
            c = ws.cell(row=fr, column=ci, value=v)
            c.border = BD; c.alignment = ctr(); c.font = font(size=10)
            c.fill = fill(GREY if ci in (1, 10) else LTBLU)
            if fmt: c.number_format = fmt
        ws.row_dimensions[fr].height = 15

    ws.freeze_panes = ws["A3"]
    print(f"  Forecast sheet: {MAX_GAMES} rows")


# ── DAILY UPDATE: write only today's 5 columns into existing file ──────────
def update_slate_daily(games, date_str):
    """
    Opens F5_Daily.xlsx and writes only:
      Col C  — Away team code
      Col D  — Away SP name
      Col E  — Away throws  (from pipeline; XLOOKUP will also resolve this)
      Col M  — Home team code
      Col N  — Home SP name
      Col O  — Home throws
    Clears rows 4–18 first, then fills from today's games.
    All formulas (xRA, bat, xRUNS, Win%, park, ump) stay untouched.
    """
    from openpyxl import load_workbook

    out = MODEL_DIR / "F5_Daily.xlsx"
    if not out.exists():
        print("  ERROR: F5_Daily.xlsx not found — run with --setup first")
        return False

    wb = load_workbook(out)
    ws = wb["Slate"]

    MAX_GAMES = 15

    # Column indices (1-based)
    # C=3 Away team, D=4 Away SP, E=5 Away throws
    # M=13 Home team, N=14 Home SP, O=15 Home throws

    AWAY_TEAM_COL  = 3
    AWAY_SP_COL    = 4
    AWAY_HAND_COL  = 5
    HOME_TEAM_COL  = 13
    HOME_SP_COL    = 14
    HOME_HAND_COL  = 15

    # Clear rows 4–18 for the 6 writable columns
    for gi in range(MAX_GAMES):
        r = 4 + gi
        for col in (AWAY_TEAM_COL, AWAY_SP_COL, AWAY_HAND_COL,
                    HOME_TEAM_COL, HOME_SP_COL, HOME_HAND_COL):
            ws.cell(row=r, column=col).value = None

    # Write today's games
    for gi, g in enumerate(games[:MAX_GAMES]):
        r = 4 + gi

        away_code  = g.get("Away Code",  g.get("Away Team",  "")) or ""
        home_code  = g.get("Home Code",  g.get("Home Team",  "")) or ""
        away_sp    = g.get("Away Pitcher", "") or ""
        home_sp    = g.get("Home Pitcher", "") or ""

        # Throws come from the pipeline slate; XLOOKUP will also resolve
        # but pre-filling means it works even before XLOOKUP resolves
        away_hand  = g.get("Away Hand", "") or ""
        home_hand  = g.get("Home Hand", "") or ""

        ws.cell(row=r, column=AWAY_TEAM_COL).value = away_code
        ws.cell(row=r, column=AWAY_SP_COL).value   = away_sp
        if away_hand:
            ws.cell(row=r, column=AWAY_HAND_COL).value = away_hand
        ws.cell(row=r, column=HOME_TEAM_COL).value = home_code
        ws.cell(row=r, column=HOME_SP_COL).value   = home_sp
        if home_hand:
            ws.cell(row=r, column=HOME_HAND_COL).value = home_hand

    wb.save(out)
    print(f"  Updated Slate: {min(len(games), MAX_GAMES)} games written → F5_Daily.xlsx")
    return True


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ClutchSheets F5 Daily Workbook")
    parser.add_argument("--date",  default=None,  help="YYYY-MM-DD (default: today)")
    parser.add_argument("--setup", action="store_true",
                        help="Build full workbook from scratch (run once)")
    args = parser.parse_args()

    gd = (date.today() if not args.date
          else datetime.strptime(args.date, "%Y-%m-%d").date())
    date_str = gd.strftime("%Y-%m-%d")

    print(f"\nClutchSheets F5 Daily Workbook  |  {date_str}")
    print("=" * 50)

    games = load_slate(date_str)
    if not games:
        print(f"  WARNING: f5_slate_{date_str}.csv not found — run f5_pipeline.py first")

    out = MODEL_DIR / "F5_Daily.xlsx"

    # ── SETUP MODE: build full file ─────────────────────────────────────────
    if args.setup or not out.exists():
        if not args.setup:
            print("  F5_Daily.xlsx not found — running full setup...")

        print("Loading all data...")
        df_pitch = load_pitchers()
        df_off   = load_team_offense()
        df_park  = load_park_factors()
        df_ump   = load_umpires()
        print(f"  {len(df_pitch)} pitchers  |  {len(df_off)} teams  |  "
              f"{len(df_park)} parks  |  {len(df_ump)} umps  |  {len(games)} games")

        print("Building workbook from scratch...")
        wb = Workbook()
        wb.remove(wb.active)

        last_p  = write_pitchers(wb, df_pitch)
        last_t  = write_team_offense(wb, df_off)
        last_pf = write_park_factors(wb, df_park)
        last_u  = write_umpires(wb, df_ump)

        last_rows = {
            "pitchers": last_p,
            "teams":    last_t,
            "parks":    last_pf,
            "umps":     last_u,
        }

        write_slate(wb, games, date_str, last_rows)
        write_forecast(wb, games, date_str)

        wb.save(out)
        print(f"\nSetup complete: {out}")
        print("From now on just run:  python build_f5_daily.py")

    # ── DAILY MODE: update only today's 5 columns ──────────────────────────
    else:
        print(f"  {len(games)} games from pipeline")
        print("Updating slate...")
        update_slate_daily(games, date_str)
        print("Done.")

    print()


if __name__ == "__main__":
    main()
