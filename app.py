import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import altair as alt
from scipy.stats import nbinom

DB_FILE = "mlb_f5_forecasts.db"

# Negative binomial parameters
# r=6.7 is the fitted dispersion; lambda (mu) varies per game
NEGBIN_R = 6.7

def negbin_params(mu: float, r: float = NEGBIN_R):
    """Convert mean (mu) and dispersion (r) to scipy nbinom (n, p) params."""
    p = r / (r + mu)
    return r, p

def negbin_win_prob(mu_away: float, mu_home: float, r: float = NEGBIN_R, max_runs: int = 20) -> float:
    """P(home wins F5) using negative binomial convolution."""
    r_a, p_a = negbin_params(mu_away, r)
    r_h, p_h = negbin_params(mu_home, r)
    runs = np.arange(0, max_runs + 1)
    pmf_a = nbinom.pmf(runs, r_a, p_a)
    pmf_h = nbinom.pmf(runs, r_h, p_h)
    p_home_win = np.sum(
        pmf_h[:, None] * pmf_a[None, :] * (np.arange(max_runs + 1)[:, None] > runs[None, :])
    )
    p_tie = np.sum(pmf_h * pmf_a)
    return float(p_home_win + 0.5 * p_tie)

def negbin_pmf_table(mu: float, r: float = NEGBIN_R, max_runs: int = 12) -> pd.DataFrame:
    """Return a small PMF table for display (runs 0–max_runs)."""
    n, p = negbin_params(mu, r)
    runs = np.arange(0, max_runs + 1)
    probs = nbinom.pmf(runs, n, p)
    return pd.DataFrame({"Runs": runs, "Probability": np.round(probs, 4)})

st.set_page_config(page_title="MLB F5 Forecast Model", layout="wide")

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM forecast_archive", conn)
    conn.close()
    return df

df = load_data()

if not df.empty:
    df["date"] = df["date"].astype(str).str.strip()
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

page = st.sidebar.radio(
    "Navigation",
    ["Today's Forecasts", "Accuracy", "Calibration", "Results", "Team Ratings"]
)

def f_completed(dfin: pd.DataFrame) -> pd.DataFrame:
    if "completed5" in dfin.columns:
        return dfin[dfin["completed5"] == 1].copy()
    return dfin.copy()

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# -----------------------------
# TODAY
# -----------------------------
if page == "Today's Forecasts":
    import os, math, json as _json
    from scipy.stats import nbinom as _nbinom
    import numpy as _np

    FORECAST_FILE = "daily_forecast.csv"
    ODDS_FILE     = "daily_odds.json"

    TEAM_FULL = {
        "ARI":"Arizona Diamondbacks","ATL":"Atlanta Braves","BAL":"Baltimore Orioles",
        "BOS":"Boston Red Sox","CHC":"Chicago Cubs","CWS":"Chicago White Sox",
        "CHW":"Chicago White Sox","CIN":"Cincinnati Reds","CLE":"Cleveland Guardians",
        "COL":"Colorado Rockies","DET":"Detroit Tigers","HOU":"Houston Astros",
        "KCR":"Kansas City Royals","LAA":"Los Angeles Angels","LAD":"Los Angeles Dodgers",
        "MIA":"Miami Marlins","MIL":"Milwaukee Brewers","MIN":"Minnesota Twins",
        "NYM":"New York Mets","NYY":"New York Yankees","ATH":"Oakland Athletics",
        "OAK":"Oakland Athletics","PHI":"Philadelphia Phillies","PIT":"Pittsburgh Pirates",
        "SDP":"San Diego Padres","SEA":"Seattle Mariners","SFG":"San Francisco Giants",
        "STL":"St. Louis Cardinals","TBR":"Tampa Bay Rays","TEX":"Texas Rangers",
        "TOR":"Toronto Blue Jays","WSN":"Washington Nationals",
    }

    BOOKS = ["betonlineag","fanduel","draftkings","betmgm","betrivers"]
    BOOK_LABELS = {
        "betonlineag": "BetOnline",
        "fanduel":     "FanDuel",
        "draftkings":  "DraftKings",
        "betmgm":      "BetMGM",
        "betrivers":   "BetRivers",
    }
    BOOK_COLORS = {
        "betonlineag": "#e8a020",
        "fanduel":     "#1493ff",
        "draftkings":  "#53d337",
        "betmgm":      "#c8a84b",
        "betrivers":   "#e03030",
    }

    NEGBIN_R = 6.7

    @st.cache_data(ttl=300)
    def load_forecast_csv():
        if not os.path.exists(FORECAST_FILE):
            return pd.DataFrame()
        fc = pd.read_csv(FORECAST_FILE, encoding="utf-8-sig")
        fc["date"] = fc["date"].astype(str).str.strip()
        return fc

    @st.cache_data(ttl=300)
    def load_odds_json():
        if not os.path.exists(ODDS_FILE):
            return {}
        with open(ODDS_FILE,"r",encoding="utf-8") as f:
            return _json.load(f)

    def match_odds(games_dict, away_code, home_code):
        away_full = TEAM_FULL.get(away_code,"").lower()
        home_full = TEAM_FULL.get(home_code,"").lower()
        for key, data in games_dict.items():
            if away_full in data.get("away","").lower() and home_full in data.get("home","").lower():
                return data
        return None

    def fmt_ml(val):
        if val is None: return "<span class='dash'>\u2014</span>"
        cls = "pos" if val > 0 else "neg"
        txt = f"+{val}" if val > 0 else str(val)
        return f"<span class='{cls}'>{txt}</span>"

    def fmt_ou(val):
        if val is None: return "<span class='dash'>\u2014</span>"
        cls = "pos" if val > 0 else "neg"
        txt = f"+{val}" if val > 0 else str(val)
        return f"<span class='{cls} ou-pr'>{txt}</span>"

    def prob_to_american(p):
        if p is None or (isinstance(p,float) and math.isnan(p)): return ""
        p = max(min(p,0.9999),0.0001)
        odds = round(-(p/(1-p))*100) if p >= 0.5 else round(((1-p)/p)*100)
        return f"+{odds}" if odds > 0 else str(odds)

    def fv_html(fv_str):
        if not fv_str: return "<span class='dash'>\u2014</span>"
        cls = "pos" if fv_str.startswith("+") else "neg"
        return f"<span class='{cls} bold'>{fv_str}</span>"

    def total_fv(mu_total, r=NEGBIN_R):
        if mu_total is None or (isinstance(mu_total,float) and math.isnan(mu_total)):
            return "", "", ""
        line = round(mu_total * 2) / 2
        if mu_total <= line:
            line -= 0.5
        line = max(line, 0.5)
        n_combined = r * 2
        p_nb = n_combined / (n_combined + mu_total)
        # P(under) = P(X <= floor(line)), P(over) = P(X >= ceil(line))
        # For half lines this is exact — no ties possible
        cutoff = int(math.ceil(line)) - 1
        p_under = float(_nbinom.cdf(cutoff, n_combined, p_nb))
        p_over  = 1 - p_under
        if p_over <= 0 or p_under <= 0:
            return f"{line:.1f}", "", ""
        return f"{line:.1f}", prob_to_american(p_over), prob_to_american(p_under)

    csv_df    = load_forecast_csv()
    odds_data = load_odds_json()

    if csv_df.empty:
        st.warning("No forecast file found.")
        st.stop()

    forecast_date = csv_df["date"].max()
    st.title(f"MLB F5 Forecasts \u2014 {forecast_date}")
    odds_date = odds_data.get("date","")
    st.caption(
        f"Projections from today\u2019s model run. "
        + (f"Odds fetched: {odds_date}. " if odds_date else "")
        + "Results scored overnight."
    )

    today = csv_df[csv_df["date"] == forecast_date].copy()
    today.dropna(how="all", inplace=True)
    today = today[today["away_team"].notna() & today["home_team"].notna()]

    if "away_sp" not in today.columns: today["away_sp"] = ""
    if "home_sp" not in today.columns: today["home_sp"] = ""

    today["lambda_away"] = to_num(today["lambda_away"]).round(3)
    today["lambda_home"] = to_num(today["lambda_home"]).round(3)
    today["total_proj"]  = to_num(today["total_proj"]).round(3)

    raw_wp_home = to_num(today["winprob_home"])
    raw_wp_away = 1 - raw_wp_home
    today["winprob_away"] = (raw_wp_away * 100).round(1)
    today["winprob_home"] = (raw_wp_home * 100).round(1)
    today["fv_away"] = raw_wp_away.apply(prob_to_american)
    today["fv_home"] = raw_wp_home.apply(prob_to_american)
    today = today.sort_values("winprob_home", ascending=False).reset_index(drop=True)

    games_dict = odds_data.get("games", {})
    has_odds   = bool(games_dict)

    # ── Book header cells ──────────────────────────────────────────
    book_th = ""
    if has_odds:
        for bk in BOOKS:
            col = BOOK_COLORS.get(bk, "#666")
            lbl = BOOK_LABELS.get(bk, bk)
            book_th += (
                f'<th class="bk-head" colspan="2" style="border-top:4px solid {col}">'
                f'{lbl}</th>'
            )

    # ── Devig + edge helpers ───────────────────────────────────────
    def american_to_prob(odds):
        if odds is None: return None
        return 100/(odds+100) if odds > 0 else (-odds)/(-odds+100)

    def devig_additive(p1, p2):
        if p1 is None or p2 is None: return None, None
        hold = p1 + p2
        if hold <= 0: return None, None
        return p1/hold, p2/hold

    def ml_edge(model_p, book_this, book_other):
        if None in (model_p, book_this, book_other): return None
        p1 = american_to_prob(book_this)
        p2 = american_to_prob(book_other)
        dv, _ = devig_additive(p1, p2)
        if dv is None: return None
        return (model_p - dv) * 100

    def tot_edge(p_over_model, over_odds, under_odds, side):
        if None in (p_over_model, over_odds, under_odds): return None
        p_ov = american_to_prob(over_odds)
        p_un = american_to_prob(under_odds)
        dv_ov, _ = devig_additive(p_ov, p_un)
        if dv_ov is None: return None
        edge = (p_over_model - dv_ov) * 100
        return edge if side == "over" else -edge

    EDGE_MIN = 3.0

    def ml_cell(book_odds, model_p, other_odds):
        if book_odds is None:
            return "<td class='bk-ml empty'></td>"
        txt = f"+{book_odds}" if book_odds > 0 else str(book_odds)
        edge = ml_edge(model_p, book_odds, other_odds)
        if edge is not None and edge >= EDGE_MIN:
            return (
                f'<td class="bk-ml edge-cell">'
                f'<span class="edge-odds">{txt}</span>'
                f'<span class="edge-tag">+{edge:.1f}%</span>'
                f'</td>'
            )
        cls = "pos" if book_odds > 0 else "neg"
        return f'<td class="bk-ml"><span class="{cls}">{txt}</span></td>'

    def tot_cell(ou_odds, other_odds, line, side, p_over_model):
        if ou_odds is None or line is None:
            return "<td class='bk-tot empty'></td>"
        ls   = f"{line}"
        txt  = f"+{ou_odds}" if ou_odds > 0 else str(ou_odds)
        pre  = "o" if side == "over" else "u"
        edge = tot_edge(p_over_model, 
                        ou_odds if side=="over" else other_odds,
                        other_odds if side=="over" else ou_odds,
                        side)
        if edge is not None and edge >= EDGE_MIN:
            return (
                f'<td class="bk-tot edge-cell">'
                f'<span class="tot-lbl">{pre}{ls}</span>'
                f'<span class="edge-odds-sm">{txt}</span>'
                f'<span class="edge-tag">+{edge:.1f}%</span>'
                f'</td>'
            )
        cls = "pos" if ou_odds > 0 else "neg"
        return (
            f'<td class="bk-tot">'
            f'<span class="tot-lbl">{pre}{ls}</span>'
            f'<span class="{cls} ou-pr">{txt}</span>'
            f'</td>'
        )

    # ── Row builder ────────────────────────────────────────────────
    rows_html = ""
    for idx, (_, r) in enumerate(today.iterrows()):
        away_code   = r["away_team"]
        home_code   = r["home_team"]
        away_full   = TEAM_FULL.get(away_code, away_code)
        home_full   = TEAM_FULL.get(home_code, home_code)
        away_sp     = str(r.get("away_sp","") or "").strip()
        home_sp     = str(r.get("home_sp","") or "").strip()
        mu_total    = r["total_proj"]
        model_p_away = float(r["winprob_away"]) / 100
        model_p_home = float(r["winprob_home"]) / 100

        game     = match_odds(games_dict, away_code, home_code) if has_odds else None
        gametime = (game or {}).get("commence_edt", "")

        # Total FV via NegBin
        fv_line, fv_over_str, fv_under_str = total_fv(mu_total)
        fv_total_away = (
            f'<span class="tot-lbl">o{fv_line}</span>\u00a0{fv_html(fv_over_str)}'
            if fv_line else "\u2014"
        )
        fv_total_home = (
            f'<span class="tot-lbl">u{fv_line}</span>\u00a0{fv_html(fv_under_str)}'
            if fv_line else "\u2014"
        )

        # P(over) for edge calc
        fv_over_p = None
        if fv_line:
            n_c  = NEGBIN_R * 2
            p_nb = n_c / (n_c + mu_total)
            fv_over_p = 1 - float(_nbinom.cdf(int(math.ceil(float(fv_line))) - 1, n_c, p_nb))

        row_bg = "row-odd" if idx % 2 == 0 else "row-even"

        # Book cells
        away_bk = home_bk = ""
        if has_odds:
            for bk in BOOKS:
                ml   = (game or {}).get("ml",    {}).get(bk, {})
                tot  = (game or {}).get("total", {}).get(bk, {})
                a_ml  = ml.get("away")
                h_ml  = ml.get("home")
                over  = tot.get("over")
                under = tot.get("under")
                line  = tot.get("line")
                away_bk += ml_cell(a_ml, model_p_away, h_ml)
                away_bk += tot_cell(over, under, line, "over", fv_over_p)
                home_bk += ml_cell(h_ml, model_p_home, a_ml)
                home_bk += tot_cell(under, over, line, "under", fv_over_p)

        rows_html += (
            f'<tr class="ar {row_bg}">'
            f'<td class="tc">'
            f'<span class="side-tag away-tag">Away</span>'
            f'<span class="tm">{away_full}</span>'
            + (f'<span class="sp-name">{away_sp}</span>' if away_sp else '')
            + f'</td>'
            f'<td class="time-cell" rowspan="2">'
            f'<span class="gt">{gametime}</span>'
            f'<span class="edt">EDT</span>'
            f'</td>'
            f'<td class="nc pr">{r["lambda_away"]:.2f}</td>'
            f'<td class="nc totc" rowspan="2">'
            f'<span class="tot-num">{mu_total:.2f}</span>'
            f'<span class="xtot-lbl">xT</span>'
            f'</td>'
            f'<td class="nc wc">{r["winprob_away"]:.1f}%</td>'
            f'<td class="nc fvc model-div">{fv_html(r["fv_away"])}</td>'
            f'<td class="nc fv-tot">{fv_total_away}</td>'
            f'{away_bk}</tr>'
            f'<tr class="hr {row_bg}">'
            f'<td class="tc">'
            f'<span class="side-tag home-tag">Home</span>'
            f'<span class="tm">{home_full}</span>'
            + (f'<span class="sp-name">{home_sp}</span>' if home_sp else '')
            + f'</td>'
            f'<td class="nc pr">{r["lambda_home"]:.2f}</td>'
            f'<td class="nc wc">{r["winprob_home"]:.1f}%</td>'
            f'<td class="nc fvc model-div">{fv_html(r["fv_home"])}</td>'
            f'<td class="nc fv-tot">{fv_total_home}</td>'
            f'{home_bk}</tr>'
            f'<tr class="gr"><td colspan="99"></td></tr>'
        )

    # ── CSS ────────────────────────────────────────────────────────
    css = (
        "* {box-sizing:border-box;margin:0;padding:0}"
        "body {background:#0d1117;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
        ".t {width:100%;border-collapse:collapse;font-size:0.82rem;color:#bbb}"

        # Header
        ".t thead tr {border-bottom:3px solid #30363d}"
        ".t th {padding:9px 10px;color:#c9d1d9;font-size:0.62rem;font-weight:700;"
            "text-transform:uppercase;letter-spacing:0.07em;text-align:right;"
            "white-space:nowrap;background:#161b22;border-bottom:3px solid #30363d}"
        ".t th.la {text-align:left}"

        # Model section divider
        ".model-div {border-left:3px solid #388bfd !important}"

        # Book headers
        ".bk-head {border-left:3px solid #30363d;text-align:center !important;"
            "font-size:0.66rem !important;letter-spacing:0.05em !important;"
            "padding:9px 8px !important;color:#c9d1d9 !important;background:#161b22 !important}"

        # Data rows
        ".row-odd  td {background:#0d1117}"
        ".row-even td {background:#111620}"
        ".ar td {padding:5px 10px 1px 10px;border-top:1px solid #21262d;vertical-align:middle}"
        ".hr td {padding:1px 10px 8px 10px;vertical-align:middle}"
        ".gr td {height:2px;background:#0d1117;border:none}"

        # Team cell
        ".tc {white-space:nowrap;min-width:195px;text-align:left;padding-left:10px !important}"
        ".side-tag {display:inline-block;font-size:0.52rem;font-weight:700;letter-spacing:0.08em;"
            "padding:1px 4px;border-radius:2px;margin-right:6px;vertical-align:middle;"
            "text-transform:uppercase}"
        ".away-tag {background:#0d2f4f;color:#58a6ff}"
        ".home-tag {background:#0f2d1a;color:#3fb950}"
        ".tm {color:#e6edf3;font-weight:500;font-size:0.85rem;vertical-align:middle}"
        ".sp-name {display:block;color:#484f58;font-size:0.70rem;margin-top:0px;"
            "margin-left:42px}"

        # Time cell
        ".time-cell {text-align:center;white-space:nowrap;min-width:58px;"
            "vertical-align:middle !important}"
        ".gt {display:block;color:#8b949e;font-size:0.80rem;font-weight:600}"
        ".edt {display:block;font-size:0.58rem;color:#484f58;margin-top:1px}"

        # Model cols
        ".nc {text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}"
        ".pr {color:#e6edf3;min-width:52px;font-size:0.85rem;font-weight:700}"
        ".totc {vertical-align:middle !important;min-width:52px;text-align:right !important;"
            "padding:2px 10px !important}"
        ".tot-num {display:block;font-size:1.0rem;font-weight:700;color:#e6edf3}"
        ".xtot-lbl {display:block;font-size:0.54rem;color:#484f58;letter-spacing:0.06em;"
            "text-transform:uppercase;margin-top:1px}"
        ".wc {color:#8b949e;min-width:46px;font-size:0.80rem}"
        ".fvc {min-width:54px;font-weight:700;font-size:0.85rem}"
        ".fv-tot {min-width:90px;font-size:0.78rem;text-align:right}"
        ".tot-lbl {color:#484f58;font-size:0.74rem;margin-right:2px}"

        # Odds colours
        ".pos {color:#e6edf3;font-weight:500}"
        ".neg {color:#e6edf3;font-weight:500}"
        ".dash {color:#30363d}"
        ".bold {font-weight:700}"
        ".ou-pr {font-size:0.76rem}"

        # Book cols
        ".bk-ml {border-left:3px solid #21262d;min-width:54px;text-align:right;"
            "font-variant-numeric:tabular-nums;font-size:0.83rem;white-space:nowrap;"
            "padding-right:6px !important}"
        ".bk-tot {min-width:86px;text-align:left;font-size:0.74rem;"
            "padding-left:6px !important;white-space:nowrap}"
        ".empty {opacity:0}"

        # Edge highlighting — green cell glow
        ".edge-cell {background:#0f2d1a !important;position:relative}"
        ".edge-odds {color:#3fb950;font-weight:700;font-size:0.85rem}"
        ".edge-odds-sm {color:#3fb950;font-weight:700;font-size:0.78rem;margin:0 2px}"
        ".edge-tag {display:block;font-size:0.58rem;color:#56d364;font-weight:700;"
            "letter-spacing:0.04em;margin-top:1px}"
    )

    base_th = (
        "<th class='la' style='min-width:200px'>Team / Pitcher</th>"
        "<th style='text-align:center'>Time</th>"
        "<th>Proj Runs</th>"
        "<th style='text-align:center'>xTotal</th>"
        "<th>Win %</th>"
        "<th class='model-div'>ML FV</th>"
        "<th>Tot FV</th>"
    )

    html_table = (
        "<!DOCTYPE html><html><head>"
        f"<style>{css}</style>"
        "</head><body>"
        f"<table class='t'><thead><tr>{base_th}{book_th}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
        "</body></html>"
    )

    st.caption(
        "ℹ️ **Edge %** is calculated using full additive devig on both sides of the book line. "
        "Green cells = model edge ≥ 3% after removing the hold. "
        "**ML FV** and **Tot FV** are no-vig fair values derived from your NegBin model. "
        "Tot FV line = nearest 0.5; juice reflects P(over) vs P(under) from the run distribution."
    )
    import streamlit.components.v1 as components
    components.html(html_table, height=max(len(today) * 65 + 55, 400), scrolling=False)

    st.divider()
    st.subheader("Run Distribution (Negative Binomial, r=6.7)")
    st.caption(
        "Win probabilities above are derived from a negative binomial distribution (r=6.7) "
        "rather than Poisson. NegBin allows for overdispersion — more variance than Poisson "
        "at the same mean — which better fits real F5 run scoring. "
        "Select a game below to inspect its projected run PMF."
    )

    raw_today = csv_df[csv_df["date"] == forecast_date].copy()
    raw_today["lambda_away"] = to_num(raw_today["lambda_away"])
    raw_today["lambda_home"] = to_num(raw_today["lambda_home"])
    raw_today["winprob_home"] = to_num(raw_today["winprob_home"])

    game_labels = [
        f"{r['away_team']} @ {r['home_team']}"
        for _, r in raw_today.iterrows()
    ]
    sel_game = st.selectbox("Select game", game_labels)
    sel_row = raw_today.iloc[game_labels.index(sel_game)]

    mu_a = float(sel_row["lambda_away"])
    mu_h = float(sel_row["lambda_home"])
    wp   = float(sel_row["winprob_home"])

    pmf_a = negbin_pmf_table(mu_a)
    pmf_h = negbin_pmf_table(mu_h)

    # Verify negbin win prob matches CSV (sanity check)
    wp_check = negbin_win_prob(mu_a, mu_h)

    col_a, col_h = st.columns(2)

    with col_a:
        st.markdown(f"**{sel_row['away_team']} (Away)** — λ = {mu_a:.3f}")
        chart_a = (
            alt.Chart(pmf_a)
            .mark_bar(color="#4A90D9")
            .encode(
                x=alt.X("Runs:O", title="F5 Runs"),
                y=alt.Y("Probability:Q", title="Probability", axis=alt.Axis(format=".3f")),
                tooltip=["Runs", alt.Tooltip("Probability:Q", format=".4f")]
            )
            .properties(height=220)
        )
        st.altair_chart(chart_a, use_container_width=True)

    with col_h:
        st.markdown(f"**{sel_row['home_team']} (Home)** — λ = {mu_h:.3f}")
        chart_h = (
            alt.Chart(pmf_h)
            .mark_bar(color="#E07B39")
            .encode(
                x=alt.X("Runs:O", title="F5 Runs"),
                y=alt.Y("Probability:Q", title="Probability", axis=alt.Axis(format=".3f")),
                tooltip=["Runs", alt.Tooltip("Probability:Q", format=".4f")]
            )
            .properties(height=220)
        )
        st.altair_chart(chart_h, use_container_width=True)

    st.caption(
        f"CSV win prob: **{wp:.3f}** — NegBin recalc: **{wp_check:.3f}** "
        f"({'✓ match' if abs(wp - wp_check) < 0.01 else '⚠ mismatch >1% — check upstream r or home field adj'})"
    )

# -----------------------------
# ACCURACY
# -----------------------------
elif page == "Accuracy":
    st.title("Model Accuracy")

    dv = f_completed(df)
    if dv.empty:
        st.info("No completed-through-5 rows available yet.")
        st.stop()

    n = len(dv)
    avg_mae = to_num(dv["mae_total"]).mean()
    avg_brier = to_num(dv["brier"]).mean()
    avg_logloss = to_num(dv["logloss"]).mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample Size", n)
    c2.metric("Avg F5 Total MAE", f"{avg_mae:.3f}")
    c3.metric("Avg Brier", f"{avg_brier:.3f}", delta=f"{avg_brier - 0.25:.3f}")
    c4.metric("Avg Log Loss", f"{avg_logloss:.3f}", delta=f"{avg_logloss - 0.693:.3f}")
    st.caption("Baselines: Brier ≈ 0.25 (coin flip), Log Loss ≈ 0.693 (coin flip).")

    if "created_at" in dv.columns and dv["created_at"].notna().any():
        dv = dv.copy()

        # Use game date, not timestamp
        dv["date"] = pd.to_datetime(dv["date"])

        daily = dv.groupby("date").agg({
            "mae_total": "mean",
            "brier": "mean",
            "logloss": "mean"
        }).reset_index()

        daily = daily.sort_values("date")

        daily["rolling_mae_7"] = daily["mae_total"].rolling(7).mean()
        daily["rolling_brier_7"] = daily["brier"].rolling(7).mean()
        daily["rolling_logloss_7"] = daily["logloss"].rolling(7).mean()

        st.subheader("Rolling 7-Game Metrics")
        st.line_chart(
            daily.set_index("date")[[
                "rolling_mae_7",
                "rolling_brier_7",
                "rolling_logloss_7"
            ]]
        )
    else:
        st.info("Rolling metrics will appear once you have timestamps and more rows.")

# -----------------------------
# CALIBRATION
# -----------------------------
elif page == "Calibration":
    st.title("Win Probability Calibration")
    st.caption(
        "Win probabilities are derived from a Negative Binomial distribution (r=6.7, "
        "fitted dispersion parameter). NegBin produces wider tails than Poisson at the "
        "same mean, better reflecting game-to-game variance in F5 run scoring. "
        "Calibration checks whether those probabilities track real-world outcomes."
    )

    dc = f_completed(df)
    if dc.empty:
        st.info("No completed-through-5 rows available yet.")
        st.stop()

    dc["winprob_home"] = to_num(dc["winprob_home"])
    dc["home_win_f5"] = to_num(dc["home_win_f5"])

    bins = [0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]
    dc["bucket"] = pd.cut(dc["winprob_home"], bins=bins, include_lowest=True)

    cal = dc.groupby("bucket", dropna=False).agg(
        avg_pred=("winprob_home", "mean"),
        actual_win_rate=("home_win_f5", "mean"),
        sample_size=("home_win_f5", "count")
    ).reset_index()

    cal["bucket_label"] = cal["bucket"].astype(str)

    table = cal[["bucket_label", "avg_pred", "actual_win_rate", "sample_size"]].copy()
    table["avg_pred"] = to_num(table["avg_pred"]).round(3)
    table["actual_win_rate"] = to_num(table["actual_win_rate"]).round(3)
    st.dataframe(table, use_container_width=True, hide_index=True)

    cal_plot = cal[cal["sample_size"] > 0].copy()
    if cal_plot.empty:
        st.info("Not enough data yet to plot calibration.")
        st.stop()

    cal_plot["midpoint"] = cal_plot["bucket"].apply(lambda x: (x.left + x.right) / 2)

    points = alt.Chart(cal_plot).mark_circle(size=120).encode(
        x=alt.X("midpoint:Q", title="Predicted Probability"),
        y=alt.Y("actual_win_rate:Q", title="Actual Win Rate"),
        tooltip=[
            alt.Tooltip("avg_pred:Q", format=".3f"),
            alt.Tooltip("actual_win_rate:Q", format=".3f"),
            alt.Tooltip("sample_size:Q")
        ]
    )

    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line().encode(x="x:Q", y="y:Q")

    st.altair_chart(points + diag, use_container_width=True)

# -----------------------------
# RESULTS
# -----------------------------
elif page == "Results":
    st.title("Results (Forecasts + Actuals)")

    dv = f_completed(df)
    if dv.empty:
        st.info("No completed-through-5 rows available yet.")
        st.stop()

    dates = sorted(dv["date"].unique())
    default_dates = dates[-1:] if dates else []
    sel_dates = st.multiselect("Dates", dates, default=default_dates)

    teams = sorted(set(dv["away_team"].unique()).union(set(dv["home_team"].unique())))
    sel_team = st.selectbox("Team Filter", ["All"] + teams)

    out = dv.copy()
    if sel_dates:
        out = out[out["date"].isin(sel_dates)]
    if sel_team != "All":
        out = out[(out["away_team"] == sel_team) | (out["home_team"] == sel_team)]

    if "away_sp" not in out.columns:
        out["away_sp"] = ""
    if "home_sp" not in out.columns:
        out["home_sp"] = ""

    out_disp = out[[
        "date",
        "away_team", "away_sp",
        "home_team", "home_sp",
        "lambda_away", "lambda_home", "total_proj", "winprob_home",
        "runs_away_thru5", "runs_home_thru5", "total_actual", "home_win_f5",
        "mae_total", "brier", "logloss"
    ]].copy()

    for c in ["lambda_away","lambda_home","total_proj","mae_total","brier","logloss"]:
        out_disp[c] = to_num(out_disp[c]).round(3)
    out_disp["winprob_home"] = (to_num(out_disp["winprob_home"]) * 100).round(1)

    out_disp = out_disp.rename(columns={
        "away_team": "Away",
        "away_sp": "Away SP",
        "home_team": "Home",
        "home_sp": "Home SP",
        "lambda_away": "Proj Away Runs",
        "lambda_home": "Proj Home Runs",
        "total_proj": "Proj Total",
        "winprob_home": "Home Win %",
        "runs_away_thru5": "Act Away Runs",
        "runs_home_thru5": "Act Home Runs",
        "total_actual": "Act Total",
        "home_win_f5": "Home Result",
        "mae_total": "MAE Total",
        "brier": "Brier",
        "logloss": "Log Loss"
    })

    st.dataframe(
        out_disp.sort_values(["date", "Home Win %"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True
    )

    csv = out_disp.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, file_name="f5_results_export.csv", mime="text/csv")

# -----------------------------
# TEAM RATINGS
# -----------------------------
elif page == "Team Ratings":
    st.title("Team F5 Ratings (Model-Implied)")

    dv = f_completed(df)
    if dv.empty:
        st.info("No completed-through-5 rows available yet.")
        st.stop()

    away_rows = pd.DataFrame({
        "team": dv["away_team"],
        "proj_runs_for": to_num(dv["lambda_away"]),
        "proj_runs_against": to_num(dv["lambda_home"]),
    })

    home_rows = pd.DataFrame({
        "team": dv["home_team"],
        "proj_runs_for": to_num(dv["lambda_home"]),
        "proj_runs_against": to_num(dv["lambda_away"]),
    })

    team_df = pd.concat([away_rows, home_rows], ignore_index=True).dropna()

    ratings = team_df.groupby("team").agg(
        games=("team", "count"),
        proj_runs_for=("proj_runs_for", "mean"),
        proj_runs_against=("proj_runs_against", "mean"),
    ).reset_index()

    ratings["net_proj_runs"] = ratings["proj_runs_for"] - ratings["proj_runs_against"]

    max_games = int(ratings["games"].max()) if not ratings.empty else 1

    if max_games <= 1:
        min_games = 1
        st.info("Each team has only 1 game in archive. Add more days for filtering.")
    else:
        min_games = st.slider("Minimum games in archive", 1, max_games, 1)

    ratings = ratings[ratings["games"] >= min_games].copy()

    for c in ["proj_runs_for", "proj_runs_against", "net_proj_runs"]:
        ratings[c] = ratings[c].round(3)

    st.subheader("Net Projected F5 Runs (For - Against)")
    st.dataframe(
        ratings.sort_values("net_proj_runs", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Offenses (Projected F5 Runs For)")
        st.dataframe(
            ratings.sort_values("proj_runs_for", ascending=False).head(10),
            use_container_width=True,
            hide_index=True
        )
    with c2:
        st.subheader("Top Defenses (Lowest Projected F5 Runs Against)")
        st.dataframe(
            ratings.sort_values("proj_runs_against", ascending=True).head(10),
            use_container_width=True,
            hide_index=True
        )
