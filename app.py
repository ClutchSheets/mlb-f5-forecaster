import json
import os
import sqlite3
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ClutchSheets F5 Forecaster", layout="wide")

DAILY_ODDS_FILE = os.environ.get("DAILY_ODDS_FILE", "daily_odds.json")
DB_PATH = "mlb_f5_forecasts.db"

BOOKS = ["betonlineag", "fanduel", "draftkings", "betmgm", "betrivers"]
BOOK_LABELS = {
    "betonlineag": "BetOnline",
    "fanduel": "FanDuel",
    "draftkings": "DraftKings",
    "betmgm": "BetMGM",
    "betrivers": "BetRivers",
}
EDGE_THRESHOLD = 5.0

NAME_TO_CODE = {
    "Pittsburgh Pirates": "PIT", "Chicago Cubs": "CHC",
    "Oakland Athletics": "ATH", "Athletics": "ATH", "New York Mets": "NYM",
    "Texas Rangers": "TEX", "Los Angeles Dodgers": "LAD",
    "Los Angeles Angels": "LAA", "Cincinnati Reds": "CIN",
    "Washington Nationals": "WSN", "Milwaukee Brewers": "MIL",
    "Arizona Diamondbacks": "ARI", "Philadelphia Phillies": "PHI",
    "Minnesota Twins": "MIN", "Toronto Blue Jays": "TOR",
    "Colorado Rockies": "COL", "San Diego Padres": "SDP",
    "Miami Marlins": "MIA", "Detroit Tigers": "DET",
    "New York Yankees": "NYY", "Tampa Bay Rays": "TBR",
    "Chicago White Sox": "CHW", "Kansas City Royals": "KCR",
    "San Francisco Giants": "SFG", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "St. Louis Cardinals": "STL",
    "Cleveland Guardians": "CLE", "Atlanta Braves": "ATL",
    "Houston Astros": "HOU", "Seattle Mariners": "SEA",
}

def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def prob_to_american(p: float) -> int:
    p = max(min(p, 0.9999), 0.0001)
    if p >= 0.5:
        return -round(p / (1 - p) * 100)
    return round((1 - p) / p * 100)

def no_vig_prob(odds_a: int, odds_b: int):
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total

def edge_pct(model_p: float, market_p: float) -> float:
    return (model_p - market_p) * 100

def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1

def adj_kelly(model_p: float, decimal_odds: float, scale: float = 10) -> float:
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    return ((b * model_p - (1 - model_p)) / b) * scale

def fmt_american(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)

def highlight_color(edge: float) -> str | None:
    if edge < EDGE_THRESHOLD:
        return None
    if edge < 7.0:
        return "#2e7d32"
    if edge < 10.0:
        return "#1f6f2d"
    return "#145a22"

def format_highlight(odds: int, edge: float, color: str) -> str:
    odds_str = fmt_american(odds)
    return f"<span class='edge' style='background:{color}'>{odds_str}<small>Edge {edge:+.1f}%</small></span>"

@st.cache_data(ttl=300)
def fetch_odds():
    if not os.path.exists(DAILY_ODDS_FILE):
        return {}
    try:
        with open(DAILY_ODDS_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    lookup = {}
    for _, game in payload.get("games", {}).items():
        away_name = game.get("away", "")
        home_name = game.get("home", "")
        away_code = NAME_TO_CODE.get(away_name, away_name[:3].upper())
        home_code = NAME_TO_CODE.get(home_name, home_name[:3].upper())
        lookup[f"{away_code}@{home_code}"] = game
    return lookup

@st.cache_data(ttl=300)
def load_forecast(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = parsed.dt.strftime("%Y-%m-%d")
    return df

def load_history(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM forecast_archive ORDER BY date DESC", conn)
    conn.close()
    return df

@st.cache_data(ttl=300)
def compute_edge_clv(db_path: str, threshold: float = 0.05) -> pd.DataFrame:
    """
    Compute edge/CLV at runtime directly from forecast_archive + odds_snapshot.
    Picks the best available odds (highest edge) per game across all books.
    Returns one row per qualifying game.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        fa = pd.read_sql("""
            SELECT date, away_team, home_team, winprob_home, home_win_f5, close_home_prob_devig
            FROM forecast_archive
            WHERE completed5=1 AND close_home_prob_devig IS NOT NULL
        """, conn)
        os_ = pd.read_sql("""
            SELECT date, away_team, home_team, side, price_american, price_decimal, book, captured_at
            FROM odds_snapshot WHERE snapshot_type='6h'
        """, conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

    if fa.empty or os_.empty:
        return pd.DataFrame()

    fa["date"] = pd.to_datetime(fa["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    os_["date"] = pd.to_datetime(os_["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    os_["captured_at"] = pd.to_datetime(os_["captured_at"], errors="coerce", utc=True)

    fa["home_win_prob"] = pd.to_numeric(fa["winprob_home"], errors="coerce")
    fa["away_win_prob"] = 1 - fa["home_win_prob"]
    fa["home_result"] = pd.to_numeric(fa["home_win_f5"], errors="coerce")
    fa["close_home_prob"] = pd.to_numeric(fa["close_home_prob_devig"], errors="coerce")

    os_["side_clean"] = None
    os_.loc[os_["side"].str.lower() == "away", "side_clean"] = "away"
    os_.loc[os_["side"].str.lower() == "home", "side_clean"] = "home"
    os_.loc[os_["side"].str.upper() == os_["away_team"].str.upper(), "side_clean"] = "away"
    os_.loc[os_["side"].str.upper() == os_["home_team"].str.upper(), "side_clean"] = "home"
    os_["price_american"] = pd.to_numeric(os_["price_american"], errors="coerce")
    os_["price_decimal"] = pd.to_numeric(os_["price_decimal"], errors="coerce")
    os_ = os_.dropna(subset=["side_clean", "price_american", "price_decimal"])
    os_["captured_at"] = pd.to_datetime(os_["captured_at"], errors="coerce", utc=True).fillna(
        pd.Timestamp("1970-01-01", tz="UTC")
    )

    if os_.empty:
        return pd.DataFrame()

    df = fa.merge(os_, on=["date", "away_team", "home_team"], how="inner")
    if df.empty:
        return pd.DataFrame()

    df["model_prob"] = df.apply(
        lambda r: r["away_win_prob"] if r["side_clean"] == "away" else r["home_win_prob"], axis=1
    )
    df["close_prob"] = df.apply(
        lambda r: 1 - r["close_home_prob"] if r["side_clean"] == "away" else r["close_home_prob"], axis=1
    )
    df["result"] = df.apply(
        lambda r: 1 - r["home_result"] if r["side_clean"] == "away" else r["home_result"], axis=1
    )
    df["taken_implied"] = df["price_american"].apply(american_to_prob)
    df["edge"] = df["model_prob"] - df["taken_implied"]
    df["clv"] = df["close_prob"] - (1 / df["price_decimal"])
    df["beat_close"] = (df["clv"] > 0).astype(int)

    # One row per game: best odds (highest edge) across all books
    df["game_key"] = df["date"] + "_" + df["away_team"] + "_" + df["home_team"]
    bets = df[df["edge"] >= threshold].copy()
    bets = (
        bets.sort_values(["game_key", "edge", "captured_at"], ascending=[True, False, False])
        .drop_duplicates("game_key")
        .copy()
    )

    bets["pick_side"] = bets["side_clean"].str.upper()
    bets["odds_taken_american"] = bets["price_american"].astype(int)
    bets["close_prob_devig"] = bets["close_prob"]

    return bets.sort_values(["date", "away_team", "home_team"], ascending=[False, True, True]).reset_index(drop=True)

def render_edge_clv_section(db_path: str):
    st.subheader("Beating the Market")
    st.caption("Best available odds per game (across all books) at the 6-hour snapshot. CLV = devigged Pinnacle closing implied prob minus the implied prob of the odds taken.")

    bets = compute_edge_clv(db_path, threshold=0.05)

    if bets.empty:
        st.info("No qualifying bets yet (need completed games with closing Pinnacle odds and a 6h odds snapshot).")
        return

    st.markdown("#### Edge ≥ 5% — Best Odds Per Game")
    disp = bets.copy()
    disp["Matchup"] = disp["away_team"] + " @ " + disp["home_team"]
    disp["Edge %"] = (disp["edge"] * 100).round(1)
    disp["Odds Taken"] = disp["odds_taken_american"].apply(fmt_american)
    disp["Book"] = disp["book"]
    disp["Taken Implied %"] = (disp["taken_implied"] * 100).round(1)
    disp["Close Fair Odds"] = disp["close_prob_devig"].apply(lambda x: fmt_american(prob_to_american(float(x))))
    disp["Close Fair %"] = (disp["close_prob_devig"] * 100).round(1)
    disp["CLV %"] = (disp["clv"] * 100).round(2)
    disp["Beat Close"] = disp["beat_close"].map({1: "✅", 0: "❌"})
    disp["Result"] = disp["result"].map({1.0: "Win", 0.0: "Loss", 0.5: "Push"}).fillna("—")
    disp["Pick"] = disp["pick_side"].fillna("—")

    detail = disp[[
        "date", "Matchup", "Pick", "Edge %", "Odds Taken", "Book",
        "Taken Implied %", "Close Fair Odds", "Close Fair %",
        "CLV %", "Beat Close", "Result"
    ]].rename(columns={"date": "Date"})

    st.dataframe(detail, use_container_width=True, hide_index=True)

def get_book_odds(book_key, game, away, home):
    entry = game.get("ml", {}).get(book_key)
    if not entry:
        return None, None
    return entry.get("away"), entry.get("home")

def get_book_metrics(model_p: float, side_odds: int, opp_odds: int):
    decimal_odds = american_to_decimal(side_odds)
    adjk = adj_kelly(model_p, decimal_odds)
    if opp_odds is not None:
        market_p, _ = no_vig_prob(side_odds, opp_odds)
    else:
        market_p = american_to_prob(side_odds)
    return edge_pct(model_p, market_p), adjk

def render_metric_cell(odds: int, edge: float, td_style: str) -> str:
    color = highlight_color(edge)
    if color is None:
        return f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{fmt_american(odds)}</span></td>"
    return f"<td style='{td_style}'>{format_highlight(odds, edge, color)}</td>"

def render_games_table(df, odds_lookup):
    BG_DARK = "#0e1117"; BG_ROW = "#1a1d24"; BG_ALT = "#14171e"; BORDER = "#2a2d35"; SUBTEXT = "#888"
    book_headers = "".join(f"<th style='padding:4px 12px;text-align:center;color:{SUBTEXT};font-size:0.75em;letter-spacing:0.05em;border-bottom:1px solid {BORDER}'>{BOOK_LABELS[b]}</th>" for b in BOOKS)
    css = f"<style>body{{margin:0;padding:0;background:transparent}}table{{width:100%;border-collapse:collapse;font-family:sans-serif;background:{BG_DARK};border-radius:8px;overflow:hidden}}th{{background:#111418;color:#ffffff;font-size:0.72em;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;padding:10px 12px;border-bottom:2px solid #3a4050}}tr.game-odd td{{background:{BG_ROW}}}tr.game-even td{{background:{BG_ALT}}}tr.game-odd:hover td,tr.game-even:hover td{{filter:brightness(1.08)}}.edge{{border-radius:4px;padding:2px 7px;white-space:nowrap;display:inline-block;font-weight:600;color:#fff;line-height:1.25}}.edge small{{display:block;font-size:0.75em;font-weight:500;opacity:0.96}}td{{padding:4px 12px;border-bottom:1px solid {BORDER}}}</style>"
    TH = "padding:10px 12px;font-size:0.72em;font-weight:700;color:#fff;letter-spacing:0.08em;text-transform:uppercase;background:#111418;border-bottom:2px solid #3a4050"
    html = css + f"<table><thead><tr><th style='{TH};text-align:center;width:30px'>#</th><th style='{TH};text-align:left'>TEAM / PITCHER</th><th style='{TH};text-align:center'>TIME</th><th style='{TH};text-align:center'>WIN %</th><th style='{TH};text-align:center'>ML FV</th><th style='{TH};text-align:center'>EDGE</th>{book_headers}</tr></thead><tbody>"
    for game_num, (_, row) in enumerate(df.iterrows(), start=1):
        away = str(row.get("away_team", "")).strip(); home = str(row.get("home_team", "")).strip()
        away_sp = str(row.get("away_sp", "")).strip(); home_sp = str(row.get("home_sp", "")).strip()
        model_home_p = float(row.get("winprob_home", 0.5)); model_away_p = 1 - model_home_p
        ml_fv_away = fmt_american(prob_to_american(model_away_p)); ml_fv_home = fmt_american(prob_to_american(model_home_p))
        key = f"{away}@{home}"; game = odds_lookup.get(key, {})
        away_display = game.get("away", away); home_display = game.get("home", home); commence_edt = game.get("commence_edt", "") or "—"
        away_cells = ""; home_cells = ""; td_style = f"padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}"
        away_best_edge = None; home_best_edge = None
        for bk_key in BOOKS:
            away_odds, home_odds = get_book_odds(bk_key, game, away, home)
            if away_odds is None or home_odds is None:
                away_cells += f"<td style='{td_style}'><span style='color:#888'>—</span></td>"; home_cells += f"<td style='{td_style}'><span style='color:#888'>—</span></td>"; continue
            away_edge, _ = get_book_metrics(model_away_p, away_odds, home_odds)
            home_edge, _ = get_book_metrics(model_home_p, home_odds, away_odds)
            away_cells += render_metric_cell(away_odds, away_edge, td_style); home_cells += render_metric_cell(home_odds, home_edge, td_style)
            away_best_edge = away_edge if away_best_edge is None or away_edge > away_best_edge else away_best_edge
            home_best_edge = home_edge if home_best_edge is None or home_edge > home_best_edge else home_best_edge
        away_edge_cell = f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{away_best_edge:+.1f}%</span></td>" if away_best_edge is not None else f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
        home_edge_cell = f"<td style='{td_style}'><span style='color:#e0e0e0;font-size:0.9em'>{home_best_edge:+.1f}%</span></td>" if home_best_edge is not None else f"<td style='{td_style}'><span style='color:#888'>—</span></td>"
        game_class = "game-odd" if game_num % 2 == 1 else "game-even"
        html += f"<tr class='{game_class}'><td style='padding:4px 6px;text-align:center;border-bottom:1px solid {BORDER};vertical-align:bottom' rowspan='2'><span style='color:#888;font-size:0.8em'>{game_num}</span></td><td style='padding:4px 12px;border-bottom:1px solid {BORDER}'><span style='font-weight:600;color:#e0e0e0'>{away_display}</span><br><span style='color:#888;font-size:0.78em'>{away_sp}</span></td><td style='padding:4px 12px;text-align:center;border-bottom:2px solid {BORDER};vertical-align:middle' rowspan='2'><span style='color:#e0e0e0;font-size:0.9em'>{commence_edt}</span></td><td style='padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}'><span style='color:#e0e0e0;font-size:0.9em'>{model_away_p*100:.1f}%</span></td><td style='padding:4px 12px;text-align:center;border-bottom:1px solid {BORDER}'><span style='color:#7ec8e3;font-size:0.9em;font-weight:600'>{ml_fv_away}</span><br><span style='color:#888;font-size:0.75em'>(PIN {game.get('pinnacle', {}).get('ml', {}).get('away','—')})</span></td>{away_edge_cell}{away_cells}</tr>"
        html += f"<tr class='{game_class}'><td style='padding:4px 12px;border-bottom:2px solid {BORDER}'><span style='font-weight:600;color:#e0e0e0'>{home_display}</span><br><span style='color:#888;font-size:0.78em'>{home_sp}</span></td><td style='padding:4px 12px;text-align:center;border-bottom:2px solid {BORDER}'><span style='color:#e0e0e0;font-size:0.9em'>{model_home_p*100:.1f}%</span></td><td style='padding:4px 12px;text-align:center;border-bottom:2px solid {BORDER}'><span style='color:#7ec8e3;font-size:0.9em;font-weight:600'>{ml_fv_home}</span><br><span style='color:#888;font-size:0.75em'>(PIN {game.get('pinnacle', {}).get('ml', {}).get('home','—')})</span></td>{home_edge_cell}{home_cells}</tr>"
    html += "</tbody></table>"
    components.html(html, height=len(df) * 104 + 60, scrolling=True)

def prepare_scored_subset(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty or "completed5" not in hist.columns:
        return pd.DataFrame()
    scored = hist.copy()
    scored["completed5"] = pd.to_numeric(scored["completed5"], errors="coerce").fillna(0)
    scored = scored[scored["completed5"] == 1].copy()
    if scored.empty:
        return scored
    for col in ["winprob_home","home_win_f5","logloss","brier","mae_total","close_logloss_devig","model_vs_close_logloss_devig","home_abs_devig_close_diff","close_home_prob_devig"]:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")
    if "winprob_home" in scored.columns:
        scored = scored[(scored["winprob_home"] > 0.25) & (scored["winprob_home"] < 0.75)].copy()
    if "winprob_home" in scored.columns and "close_home_prob_devig" in scored.columns:
        scored["prob_diff_close"] = scored["winprob_home"] - scored["close_home_prob_devig"]
        scored["abs_prob_diff_close"] = scored["prob_diff_close"].abs()
    return scored.sort_values("date").copy()

def render_beating_the_market(df: pd.DataFrame):
    st.markdown("#### Model vs Market — Am I Getting Closer?")
    if "model_vs_close_logloss_devig" not in df.columns:
        st.info("No closing-line log loss data yet.")
        return
    plot_df = df.dropna(subset=["model_vs_close_logloss_devig"]).copy()
    if plot_df.empty:
        st.info("No closing-line log loss data yet.")
        return
    plot_df = plot_df.sort_values("date").reset_index(drop=True)
    plot_df["game_num"] = range(1, len(plot_df) + 1)
    plot_df["cumulative_median"] = plot_df["model_vs_close_logloss_devig"].expanding().median()
    plot_df["cumulative_mean"] = plot_df["model_vs_close_logloss_devig"].expanding().mean()

    # Dot colour by result: win=green, loss=red, push/unknown=grey
    def result_color(r):
        if r == 1.0: return "Win"
        if r == 0.0: return "Loss"
        return "Push"
    plot_df["outcome"] = plot_df["home_win_f5"].apply(result_color) if "home_win_f5" in plot_df.columns else "Unknown"

    median_delta = plot_df["model_vs_close_logloss_devig"].median()
    mean_delta   = plot_df["model_vs_close_logloss_devig"].mean()
    outlier_drag = mean_delta - median_delta
    n = len(plot_df)
    window = min(50, n)
    recent_avg = plot_df.tail(window)["model_vs_close_logloss_devig"].median()
    trend_diff = recent_avg - median_delta

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Games Scored", n)
    c2.metric("Median Log Loss Delta", f"{median_delta:+.4f}")
    c3.metric("Mean Log Loss Delta", f"{mean_delta:+.4f}")
    c4.metric("Outlier Drag", f"{outlier_drag:+.4f}")
    c5.metric(f"Last {window} Median", f"{recent_avg:+.4f}", delta=f"{trend_diff:+.4f} vs overall", delta_color="inverse")

    # Shaded last-50 band
    shade_start = max(1, n - window + 1)
    shade_df = pd.DataFrame({"x1": [shade_start], "x2": [n]})
    shade = alt.Chart(shade_df).mark_rect(opacity=0.08, color="#378AD3").encode(
        x=alt.X("x1:Q"), x2="x2:Q"
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        strokeDash=[4, 4], color="gray", opacity=0.5
    ).encode(y="y:Q")

    color_scale = alt.Scale(
        domain=["Win", "Loss", "Push"],
        range=["#2ecc71", "#e74c3c", "#888888"]
    )
    points = alt.Chart(plot_df).mark_circle(size=40, opacity=0.65).encode(
        x=alt.X("game_num:Q", title="Game #"),
        y=alt.Y("model_vs_close_logloss_devig:Q", title="Log Loss Delta (model − close)"),
        color=alt.Color("outcome:N", scale=color_scale, legend=alt.Legend(title="Result")),
        tooltip=["date:N", "away_team:N", "home_team:N", "outcome:N",
                 alt.Tooltip("model_vs_close_logloss_devig:Q", format="+.4f", title="LL Delta")]
    )

    trend_line_median = alt.Chart(plot_df).mark_line(color="#378AD3", strokeWidth=2.5).encode(
        x="game_num:Q", y="cumulative_median:Q"
    )
    trend_line_mean = alt.Chart(plot_df).mark_line(
        color="#f0a500", strokeWidth=1.5, strokeDash=[4, 3], opacity=0.7
    ).encode(x="game_num:Q", y="cumulative_mean:Q")

    chart = (shade + zero_line + points + trend_line_mean + trend_line_median).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

def render_performance(hist: pd.DataFrame):
    st.subheader("Model vs Market Performance")
    scored = prepare_scored_subset(hist)
    if scored.empty:
        st.info("No scored games yet.")
        return

    avg_abs_prob_diff = scored["abs_prob_diff_close"].mean() if "abs_prob_diff_close" in scored.columns else np.nan
    model_logloss    = scored["logloss"].mean() if "logloss" in scored.columns else np.nan
    close_logloss    = scored["close_logloss_devig"].mean() if "close_logloss_devig" in scored.columns else np.nan

    # CLV summary for side-by-side
    clv_bets = compute_edge_clv(DB_PATH, threshold=0.05)
    n_bets        = len(clv_bets)
    avg_clv       = clv_bets["clv"].mean() if n_bets > 0 else np.nan
    beat_close_rt = clv_bets["beat_close"].mean() if n_bets > 0 else np.nan

    # P/L to win 100
    def _to_win_100(row):
        odds = row["odds_taken_american"]
        if row["result"] == 1.0: return 100.0
        if row["result"] == 0.0: return -100.0 if odds > 0 else -float(abs(odds))
        return 0.0
    pl_100 = clv_bets.apply(_to_win_100, axis=1).sum() if n_bets > 0 else np.nan

    # Seven metrics in one row
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Qualified Bets (≥5%)", n_bets if n_bets > 0 else "—")
    c2.metric("Avg CLV", f"{avg_clv*100:.2f}%" if pd.notna(avg_clv) else "—")
    c3.metric("Beat Close", f"{beat_close_rt*100:.1f}%" if pd.notna(beat_close_rt) else "—")
    c4.metric("P/L (to win $100)", f"${pl_100:+.0f}" if pd.notna(pl_100) else "—")
    c5.metric("Model Log Loss", f"{model_logloss:.4f}" if pd.notna(model_logloss) else "—")
    c6.metric("Close Log Loss", f"{close_logloss:.4f}" if pd.notna(close_logloss) else "—")
    c7.metric("Avg Prob Gap vs Close", f"{avg_abs_prob_diff:.4f}" if pd.notna(avg_abs_prob_diff) else "—")

    st.markdown("---")
    render_beating_the_market(scored)

def _negbin_r(total_proj: float) -> float:
    """Dynamic dispersion: r = MAX(3, MIN(6, 2.5 + 0.4 × total_proj)).
    Matches the Excel model formula =MAX(3,MIN(6,2.5+0.4*(lambda_away+lambda_home)))."""
    return max(3.0, min(6.0, 2.5 + 0.4 * total_proj))

def _negbin_pmf(k_vals, mu, r):
    """Negative Binomial PMF: P(X=k) given mean mu and dispersion r."""
    from scipy.stats import nbinom
    p = r / (r + mu)
    return nbinom.pmf(k_vals, r, p)

def render_calibration(hist: pd.DataFrame, forecast_df: pd.DataFrame):
    """Calibration tab: confidence buckets + NegBin distribution explorer."""
    from scipy.stats import nbinom

    st.subheader("Model Calibration — Confidence Buckets")
    st.caption("How often does the model's predicted win probability match actual outcomes? Buckets use both home and away perspectives (each game contributes two data points).")

    scored = hist.copy()
    scored["completed5"] = pd.to_numeric(scored.get("completed5", pd.Series(dtype=float)), errors="coerce").fillna(0)
    scored = scored[scored["completed5"] == 1].copy()
    for col in ["winprob_home", "home_win_f5", "lambda_away", "lambda_home"]:
        if col in scored.columns:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")

    if scored.empty or "winprob_home" not in scored.columns or "home_win_f5" not in scored.columns:
        st.info("No scored games available for calibration.")
    else:
        # Build long-form: one row per (game, side) so both perspectives are captured
        home_rows = scored[["winprob_home", "home_win_f5"]].rename(
            columns={"winprob_home": "pred_prob", "home_win_f5": "actual"}
        ).copy()
        away_rows = scored[["winprob_home", "home_win_f5"]].copy()
        away_rows["pred_prob"] = 1 - scored["winprob_home"]
        away_rows["actual"]    = 1 - scored["home_win_f5"]
        long_df = pd.concat([home_rows, away_rows], ignore_index=True)
        long_df = long_df.dropna(subset=["pred_prob", "actual"])
        # Keep only the "confident" side per game — pred_prob >= 0.50
        long_df = long_df[long_df["pred_prob"] >= 0.50].copy()

        # 5% buckets: 50-55, 55-60, 60-65, 65-70, 70+
        edges = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
        labels = ["50–55%", "55–60%", "60–65%", "65–70%", "70%+"]
        long_df["bucket"] = pd.cut(long_df["pred_prob"], bins=edges, labels=labels, right=False)
        bucket_stats = (
            long_df.groupby("bucket", observed=True)
            .agg(
                n=("actual", "count"),
                actual_win_rate=("actual", "mean"),
                avg_pred_prob=("pred_prob", "mean"),
            )
            .reset_index()
        )
        bucket_stats["delta"] = bucket_stats["actual_win_rate"] - bucket_stats["avg_pred_prob"]
        bucket_stats["midpoint"] = bucket_stats["avg_pred_prob"]

        # Summary metrics row
        total_games = len(scored)
        overall_actual = long_df["actual"].mean()
        overall_pred   = long_df["pred_prob"].mean()
        mean_abs_err   = (long_df["actual"] - long_df["pred_prob"]).abs().mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scored Games", total_games)
        c2.metric("Avg Predicted Win%", f"{overall_pred*100:.1f}%")
        c3.metric("Avg Actual Win%", f"{overall_actual*100:.1f}%")
        c4.metric("Mean Abs Calibration Error", f"{mean_abs_err*100:.2f}%")

        st.markdown("---")

        # Bucket table
        st.markdown("#### Bucket Breakdown")
        tbl = bucket_stats.copy()
        tbl["Avg Predicted"] = (tbl["avg_pred_prob"] * 100).round(1).astype(str) + "%"
        tbl["Actual Win Rate"] = (tbl["actual_win_rate"] * 100).round(1).astype(str) + "%"
        tbl["Delta (Actual − Pred)"] = tbl["delta"].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
        tbl["N"] = tbl["n"].astype(int)
        st.dataframe(
            tbl[["bucket", "N", "Avg Predicted", "Actual Win Rate", "Delta (Actual − Pred)"]].rename(columns={"bucket": "Bucket"}),
            use_container_width=True, hide_index=True
        )

        st.markdown("---")

        # Reliability diagram: dots on diagonal
        st.markdown("#### Reliability Diagram — Predicted vs Actual Win Rate")
        chart_df = bucket_stats.dropna(subset=["actual_win_rate", "avg_pred_prob"]).copy()
        chart_df["bucket_str"] = chart_df["bucket"].astype(str)
        # Dot size proportional to sample size
        chart_df["n_sqrt"] = np.sqrt(chart_df["n"]) * 18

        # Perfect calibration diagonal (x=y from 0.50 to 0.85)
        diag_df = pd.DataFrame({"x": [0.48, 0.88], "y": [0.48, 0.88]})
        diag_line = alt.Chart(diag_df).mark_line(
            color="#f0a500", strokeDash=[6, 3], strokeWidth=2, opacity=0.75
        ).encode(x=alt.X("x:Q"), y=alt.Y("y:Q"))

        # Error bars: vertical lines from predicted to actual per bucket
        error_bars = alt.Chart(chart_df).mark_rule(
            color="#888888", strokeWidth=1.5, opacity=0.5
        ).encode(
            x=alt.X("avg_pred_prob:Q"),
            y=alt.Y("avg_pred_prob:Q"),
            y2="actual_win_rate:Q"
        )

        # Dots coloured by delta direction
        chart_df["over_under"] = chart_df["delta"].apply(
            lambda d: "Overconfident" if d < -0.02 else ("Underconfident" if d > 0.02 else "Calibrated")
        )
        color_scale = alt.Scale(
            domain=["Overconfident", "Calibrated", "Underconfident"],
            range=["#e74c3c", "#2ecc71", "#378AD3"]
        )
        dots = alt.Chart(chart_df).mark_circle(opacity=0.9, stroke="#ffffff", strokeWidth=1).encode(
            x=alt.X("avg_pred_prob:Q", title="Predicted Win Probability",
                    scale=alt.Scale(domain=[0.48, 0.88]), axis=alt.Axis(format="%")),
            y=alt.Y("actual_win_rate:Q", title="Actual Win Rate",
                    scale=alt.Scale(domain=[0.20, 0.88]), axis=alt.Axis(format="%")),
            size=alt.Size("n:Q", scale=alt.Scale(range=[100, 900]), legend=alt.Legend(title="Sample (N)")),
            color=alt.Color("over_under:N", scale=color_scale, legend=alt.Legend(title="Calibration")),
            tooltip=[
                alt.Tooltip("bucket_str:N", title="Bucket"),
                alt.Tooltip("n:Q", title="N"),
                alt.Tooltip("avg_pred_prob:Q", title="Avg Predicted", format=".1%"),
                alt.Tooltip("actual_win_rate:Q", title="Actual Win Rate", format=".1%"),
                alt.Tooltip("delta:Q", title="Delta", format="+.1%"),
            ]
        )

        # Bucket labels next to each dot
        bucket_labels = alt.Chart(chart_df).mark_text(
            align="left", dx=10, dy=-4, fontSize=11, color="#cccccc"
        ).encode(
            x=alt.X("avg_pred_prob:Q"),
            y=alt.Y("actual_win_rate:Q"),
            text=alt.Text("bucket_str:N")
        )

        reliability_chart = (diag_line + error_bars + dots + bucket_labels).properties(height=380)
        st.altair_chart(reliability_chart, use_container_width=True)
        st.caption("Each dot = one confidence bucket. Size = sample N. Vertical line shows gap from predicted to actual. 🟡 Diagonal = perfect calibration. 🔴 Below diagonal = overconfident.")

    # -----------------------------------------------------------------------
    # NegBin Distribution Explorer
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### NegBin Run Distribution Explorer")
    st.caption("Distributions reconstructed from stored λ (mean) values using the pipeline's dynamic dispersion formula: r = MAX(3, MIN(6, 2.5 + 0.4 × projected total)). Win% shown is the **raw NegBin output** — expect ~8% lower than the stored final win prob due to HFA/handedness adjustments applied downstream.")

    # Build game list — prefer today's forecast, fall back to recent archived games
    game_options = []
    if not forecast_df.empty and "lambda_away" in forecast_df.columns:
        for _, row in forecast_df.iterrows():
            label = f"{row.get('date','')} | {row.get('away_team','')} @ {row.get('home_team','')} (forecast)"
            game_options.append({"label": label, "row": row, "source": "forecast"})

    if not hist.empty and "lambda_away" in hist.columns:
        recent = hist.copy()
        recent["lambda_away"] = pd.to_numeric(recent["lambda_away"], errors="coerce")
        recent["lambda_home"] = pd.to_numeric(recent["lambda_home"], errors="coerce")
        recent = recent.dropna(subset=["lambda_away", "lambda_home"]).sort_values("date", ascending=False).head(30)
        for _, row in recent.iterrows():
            result_tag = ""
            if pd.notna(row.get("home_win_f5")):
                result_tag = f" → {'HOME WIN' if row['home_win_f5']==1 else 'AWAY WIN' if row['home_win_f5']==0 else 'PUSH'}"
            label = f"{row.get('date','')} | {row.get('away_team','')} @ {row.get('home_team','')} (archived{result_tag})"
            game_options.append({"label": label, "row": row, "source": "archive"})

    if not game_options:
        st.info("No games with NegBin parameters available.")
        return

    selected_label = st.selectbox("Select game", [g["label"] for g in game_options], key="negbin_game_select")
    selected = next(g for g in game_options if g["label"] == selected_label)
    row = selected["row"]

    mu_away = float(row["lambda_away"])
    mu_home = float(row["lambda_home"])
    away_team = str(row.get("away_team", "Away"))
    home_team = str(row.get("home_team", "Home"))
    winprob_home = float(row.get("winprob_home", 0.5))
    winprob_away = 1 - winprob_home
    total_proj = mu_away + mu_home
    r = _negbin_r(total_proj)

    k_vals = np.arange(0, 12)
    pmf_away = _negbin_pmf(k_vals, mu_away, r)
    pmf_home = _negbin_pmf(k_vals, mu_home, r)

    # Joint PMF for total runs
    max_k = 12
    total_probs = {}
    for ka in range(max_k):
        for kh in range(max_k):
            t = ka + kh
            total_probs[t] = total_probs.get(t, 0) + pmf_away[ka] * pmf_home[kh]
    total_k = sorted(total_probs.keys())
    total_pmf = np.array([total_probs[k] for k in total_k])

    # Win prob from joint PMF (sanity check vs stored)
    win_home_joint = sum(
        pmf_away[ka] * pmf_home[kh]
        for ka in range(max_k) for kh in range(max_k) if kh > ka
    )
    win_away_joint = sum(
        pmf_away[ka] * pmf_home[kh]
        for ka in range(max_k) for kh in range(max_k) if ka > kh
    )
    push_joint = sum(
        pmf_away[ka] * pmf_home[kh]
        for ka in range(max_k) for kh in range(max_k) if ka == kh
    )

    # Header metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(f"{away_team} λ", f"{mu_away:.3f}")
    c2.metric(f"{home_team} λ", f"{mu_home:.3f}")
    c3.metric("Proj Total", f"{total_proj:.2f}")
    c4.metric("Dispersion r", f"{r:.2f}")
    c5.metric(f"{home_team} Win% (stored)", f"{winprob_home*100:.1f}%")
    c6.metric(f"{home_team} Win% (pre-adj)", f"{win_home_joint*100:.1f}%", delta=f"{(win_home_joint - winprob_home)*100:+.1f}% vs final", help="Raw NegBin output before pipeline adjustments (HFA, handedness). Expect ~8% lower than stored win prob for home teams.")

    # Result annotation for archived games
    if selected["source"] == "archive" and pd.notna(row.get("home_win_f5")):
        actual_away = int(row.get("runs_away_thru5", 0) or 0)
        actual_home = int(row.get("runs_home_thru5", 0) or 0)
        result_str = "HOME WIN" if row["home_win_f5"] == 1 else "AWAY WIN" if row["home_win_f5"] == 0 else "PUSH"
        st.info(f"**Result:** {away_team} {actual_away} – {actual_home} {home_team} → **{result_str}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**{away_team} Run Distribution**")
        away_df = pd.DataFrame({"runs": k_vals, "prob": pmf_away, "team": away_team})
        mode_away = int(k_vals[np.argmax(pmf_away)])
        away_chart = alt.Chart(away_df).mark_bar(opacity=0.85).encode(
            x=alt.X("runs:O", title="F5 Runs"),
            y=alt.Y("prob:Q", title="Probability", axis=alt.Axis(format=".0%")),
            color=alt.condition(
                alt.datum.runs == mode_away,
                alt.value("#f0a500"), alt.value("#e74c3c")
            ),
            tooltip=[alt.Tooltip("runs:O", title="Runs"), alt.Tooltip("prob:Q", title="P(X=k)", format=".3f")]
        ).properties(height=260)
        st.altair_chart(away_chart, use_container_width=True)
        st.caption(f"Mode: {mode_away}  |  Mean: {mu_away:.2f}  |  P(0): {pmf_away[0]:.1%}")

    with col2:
        st.markdown(f"**{home_team} Run Distribution**")
        home_df = pd.DataFrame({"runs": k_vals, "prob": pmf_home, "team": home_team})
        mode_home = int(k_vals[np.argmax(pmf_home)])
        home_chart = alt.Chart(home_df).mark_bar(opacity=0.85).encode(
            x=alt.X("runs:O", title="F5 Runs"),
            y=alt.Y("prob:Q", title="Probability", axis=alt.Axis(format=".0%")),
            color=alt.condition(
                alt.datum.runs == mode_home,
                alt.value("#f0a500"), alt.value("#1a7a3a")
            ),
            tooltip=[alt.Tooltip("runs:O", title="Runs"), alt.Tooltip("prob:Q", title="P(X=k)", format=".3f")]
        ).properties(height=260)
        st.altair_chart(home_chart, use_container_width=True)
        st.caption(f"Mode: {mode_home}  |  Mean: {mu_home:.2f}  |  P(0): {pmf_home[0]:.1%}")

    with col3:
        st.markdown("**Game Total Distribution**")
        total_df = pd.DataFrame({"runs": total_k, "prob": total_pmf})
        proj_total = mu_away + mu_home
        total_chart = alt.Chart(total_df).mark_bar(color="#378AD3", opacity=0.85).encode(
            x=alt.X("runs:O", title="F5 Total Runs"),
            y=alt.Y("prob:Q", title="Probability", axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("runs:O", title="Total"), alt.Tooltip("prob:Q", title="P(Total=k)", format=".3f")]
        ).properties(height=260)
        proj_line = alt.Chart(pd.DataFrame({"x": [proj_total]})).mark_rule(
            color="#f0a500", strokeDash=[4, 3], strokeWidth=2
        ).encode(x=alt.X("x:Q"))
        st.altair_chart(total_chart, use_container_width=True)
        st.caption(f"Proj: {proj_total:.2f}  |  P(≤4): {sum(v for k,v in zip(total_k,total_pmf) if k<=4):.1%}  |  P(≥6): {sum(v for k,v in zip(total_k,total_pmf) if k>=6):.1%}")

    # Win probability breakdown bar
    st.markdown("---")
    st.markdown("#### Win Probability Breakdown (from Joint NegBin PMF)")
    wp_df = pd.DataFrame({
        "outcome": [f"{away_team} Win", "Push/Tie", f"{home_team} Win"],
        "prob": [win_away_joint, push_joint, win_home_joint],
        "color": ["#e74c3c", "#888888", "#1a7a3a"]
    })
    wp_chart = alt.Chart(wp_df).mark_bar(size=48).encode(
        x=alt.X("prob:Q", title="Probability", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("outcome:N", sort=[f"{away_team} Win", "Push/Tie", f"{home_team} Win"], title=""),
        color=alt.Color("outcome:N", scale=alt.Scale(
            domain=[f"{away_team} Win", "Push/Tie", f"{home_team} Win"],
            range=["#e74c3c", "#888888", "#1a7a3a"]
        ), legend=None),
        tooltip=[alt.Tooltip("outcome:N"), alt.Tooltip("prob:Q", format=".2%")]
    ).properties(height=160)
    st.altair_chart(wp_chart, use_container_width=True)
    st.caption(f"Push probability: {push_joint:.2%} — included in moneyline devig but worth watching if high")


def main():
    st.markdown("<style>header[data-testid='stHeader']{display:none!important}#MainMenu{display:none!important}footer{display:none!important}.block-container{padding-top:0.5rem!important;padding-bottom:0!important}h2{margin-top:0!important}</style>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#1a7a3a;margin:0 0 6px 0'>⚾ ClutchSheets F5 Baseball Forecaster</h2>", unsafe_allow_html=True)
    CSV_PATH = os.environ.get("FORECAST_CSV", "daily_forecast.csv")
    fetched_at = None; odds_file_date = None
    if os.path.exists(DAILY_ODDS_FILE):
        try:
            with open(DAILY_ODDS_FILE, encoding="utf-8") as f:
                odds_payload = json.load(f)
            fetched_at = odds_payload.get("fetched_at"); odds_file_date = odds_payload.get("date")
        except Exception:
            pass
    tab1, tab2, tab3 = st.tabs(["📅 Today's Games", "📊 Performance", "🎯 Calibration"])
    with tab1:
        if fetched_at:
            date_line = f"<div style='color:#8f9bad;font-size:0.82em;margin-top:6px'>Odds file date: {odds_file_date}</div>" if odds_file_date else ""
            st.markdown(f"<div style='background:#0e1117;padding:12px 16px;border-radius:10px;display:inline-block;margin: 2px 0 14px 0;'><div style='color:#a7b0c0;font-size:0.75em;letter-spacing:0.08em'>LAST UPDATED</div><div style='color:#ffffff;font-size:1.12em;font-weight:700;margin-top:4px'>{fetched_at}</div>{date_line}</div>", unsafe_allow_html=True)
        df = load_forecast(CSV_PATH)
        if df.empty:
            st.warning("No forecast data found.")
        else:
            today_str = date.today().strftime("%Y-%m-%d")
            slate_df = df.copy()
            if "date" in slate_df.columns:
                exact = slate_df[slate_df["date"] == today_str].copy()
                if exact.empty:
                    latest_date = slate_df["date"].dropna().max()
                    slate_df = slate_df[slate_df["date"] == latest_date].copy() if pd.notna(latest_date) else pd.DataFrame()
                else:
                    slate_df = exact
            if not slate_df.empty:
                odds_lookup = fetch_odds()
                if not odds_lookup:
                    st.warning("⚠️ No odds data — check daily_odds.json.")
                render_games_table(slate_df, odds_lookup)
    with tab2:
        render_edge_clv_section(DB_PATH)
        st.markdown("---")
        hist = load_history(DB_PATH)
        render_performance(hist)
    with tab3:
        hist = load_history(DB_PATH)
        forecast_df = load_forecast(CSV_PATH)
        render_calibration(hist, forecast_df)

if __name__ == "__main__":
    main()
