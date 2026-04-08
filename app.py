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

    FORECAST_FILE = "daily_forecast.csv"
    ODDS_FILE     = "daily_odds.json"

    TEAM_NAME_MAP = {
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

    BOOKS = ["betonlineag","betmgm","betrivers","draftkings","fanduel"]
    BOOK_LABELS = {
        "betonlineag":"BetOnline",
        "betmgm":     "BetMGM",
        "betrivers":  "BetRivers",
        "draftkings": "DraftKings",
        "fanduel":    "FanDuel",
    }

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
        with open(ODDS_FILE, "r", encoding="utf-8") as f:
            return _json.load(f)

    def match_odds(games_dict, away_code, home_code):
        away_full = TEAM_NAME_MAP.get(away_code, "")
        home_full = TEAM_NAME_MAP.get(home_code, "")
        for key, data in games_dict.items():
            a = data.get("away", "")
            h = data.get("home", "")
            if away_full.lower() in a.lower() and home_full.lower() in h.lower():
                return data
        return None

    def fmt_odds(val):
        if val is None: return "\u2014"
        return f"+{val}" if val > 0 else str(val)

    def prob_to_american(p):
        if p is None or (isinstance(p, float) and math.isnan(p)): return ""
        p = max(min(p, 0.9999), 0.0001)
        odds = round(-(p/(1-p))*100) if p >= 0.5 else round(((1-p)/p)*100)
        return f"+{odds}" if odds > 0 else str(odds)

    def ml_color(val):
        if val is None: return "color:#444;"
        return "color:#5cb85c;font-weight:600;" if val > 0 else "color:#c0c0c0;"

    def fv_color(fv_str):
        if not fv_str: return "color:#444;"
        return "color:#5cb85c;font-weight:600;" if fv_str.startswith("+") else "color:#c0c0c0;"

    # ── Load data ──────────────────────────────────────────────────
    csv_df    = load_forecast_csv()
    odds_data = load_odds_json()

    if csv_df.empty:
        st.warning("No forecast file found. Drop daily_forecast.csv next to app.py.")
        st.stop()

    forecast_date = csv_df["date"].max()
    st.title(f"MLB F5 Forecasts \u2014 {forecast_date}")

    odds_date = odds_data.get("date", "")
    if odds_date:
        st.caption(
            f"Projections from today's model run. "
            f"Odds fetched: {odds_date}. Results scored overnight."
        )
    else:
        st.caption("Projections from today's model run. Results scored overnight.")
        st.info("\u1f4c4 No odds file found. Run 2_push_forecasts.bat to include book odds.")

    today = csv_df[csv_df["date"] == forecast_date].copy()
    today.dropna(how="all", inplace=True)
    today = today[today["away_team"].notna() & today["home_team"].notna()]

    if "away_sp" not in today.columns: today["away_sp"] = ""
    if "home_sp" not in today.columns: today["home_sp"] = ""

    today["lambda_away"] = to_num(today["lambda_away"]).round(2)
    today["lambda_home"] = to_num(today["lambda_home"]).round(2)
    today["total_proj"]  = to_num(today["total_proj"]).round(2)

    raw_wp_home = to_num(today["winprob_home"])
    raw_wp_away = 1 - raw_wp_home
    today["winprob_away"] = (raw_wp_away * 100).round(1)
    today["winprob_home"] = (raw_wp_home * 100).round(1)
    today["fv_away"] = raw_wp_away.apply(prob_to_american)
    today["fv_home"] = raw_wp_home.apply(prob_to_american)
    today = today.sort_values("winprob_home", ascending=False).reset_index(drop=True)

    games_dict = odds_data.get("games", {})
    has_odds   = bool(games_dict)

    # ── Build table ────────────────────────────────────────────────
    # ── Unabated-style compact layout ─────────────────────────────
    # Header: ML | Total per book (2 cols each)
    book_headers = ""
    if has_odds:
        for bk in BOOKS:
            book_headers += f'<th class="bk-head" colspan="2">{BOOK_LABELS[bk]}</th>'

    rows_html = ""
    for _, r in today.iterrows():
        away_code = r["away_team"]
        home_code = r["home_team"]
        total_str = f"{r['total_proj']:.1f}" if not pd.isna(r["total_proj"]) else "\u2014"
        game      = match_odds(games_dict, away_code, home_code) if has_odds else None

        away_bk = home_bk = ""
        if has_odds:
            for bk in BOOKS:
                ml  = (game or {}).get("ml",    {}).get(bk, {})
                tot = (game or {}).get("total", {}).get(bk, {})
                a_ml  = ml.get("away")
                h_ml  = ml.get("home")
                over  = tot.get("over")
                under = tot.get("under")
                line  = tot.get("line", "\u2014")
                # Away row: ML | o{line} over_price
                away_bk += (
                    f'<td class="bk-ml" style="{ml_color(a_ml)}">{fmt_odds(a_ml)}</td>'
                    f'<td class="bk-tot ov">o{line}&nbsp;<span class="ou-pr">{fmt_odds(over)}</span></td>'
                )
                # Home row: ML | u{line} under_price
                home_bk += (
                    f'<td class="bk-ml" style="{ml_color(h_ml)}">{fmt_odds(h_ml)}</td>'
                    f'<td class="bk-tot un">u{line}&nbsp;<span class="ou-pr">{fmt_odds(under)}</span></td>'
                )

        rows_html += (
            '<tr class="ar">'
            f'<td class="tc"><span class="badge ab">A</span><span class="tm">{away_code}</span></td>'
            f'<td class="sc">{r.get("away_sp","")}</td>'
            f'<td class="nc pr">{r["lambda_away"]:.2f}</td>'
            f'<td class="nc totc" rowspan="2">{total_str}</td>'
            f'<td class="nc wc">{r["winprob_away"]:.1f}%</td>'
            f'<td class="nc fvc" style="{fv_color(r["fv_away"])}">{r["fv_away"]}</td>'
            f'{away_bk}</tr>'
            '<tr class="hr">'
            f'<td class="tc"><span class="badge hb">H</span><span class="tm">{home_code}</span></td>'
            f'<td class="sc">{r.get("home_sp","")}</td>'
            f'<td class="nc pr">{r["lambda_home"]:.2f}</td>'
            f'<td class="nc wc">{r["winprob_home"]:.1f}%</td>'
            f'<td class="nc fvc" style="{fv_color(r["fv_home"])}">{r["fv_home"]}</td>'
            f'{home_bk}</tr>'
            '<tr class="gr"><td colspan="99"></td></tr>'
        )

    css = (
        "* {box-sizing:border-box;margin:0;padding:0}"
        "body {background:#0e1117;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}"
        ".t {width:100%;border-collapse:collapse;font-size:0.82rem;color:#ccc}"

        # Header
        ".t thead tr {border-bottom:2px solid #2d2d2d}"
        ".t th {padding:5px 8px;color:#555;font-size:0.62rem;font-weight:700;"
            "text-transform:uppercase;letter-spacing:0.06em;text-align:right;white-space:nowrap}"
        ".t th.la {text-align:left}"
        ".bk-head {border-left:3px solid #1e1e1e;color:#4a4a4a !important;"
            "text-align:center !important;padding:5px 4px !important}"

        # Rows — tight like Unabated
        ".ar td {padding:6px 8px 2px 8px;border-top:1px solid #1e1e1e;vertical-align:middle}"
        ".hr td {padding:2px 8px 6px 8px;vertical-align:middle}"
        ".gr td {height:3px;background:#0e1117}"

        # Team + SP
        ".tc {white-space:nowrap;min-width:62px;text-align:left}"
        ".tm {color:#e0e0e0;font-weight:600;font-size:0.84rem;letter-spacing:0.02em}"
        ".sc {color:#666;font-size:0.76rem;min-width:130px;text-align:left}"

        # Model columns
        ".nc {text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}"
        ".pr {color:#888;min-width:52px;font-size:0.80rem}"
        ".totc {vertical-align:middle !important;font-size:0.95rem;font-weight:700;"
            "color:#e0e0e0;min-width:50px;text-align:right}"
        ".wc {color:#999;min-width:46px;font-size:0.80rem}"
        ".fvc {min-width:54px;font-weight:700;font-size:0.82rem}"

        # Book columns — Unabated style
        ".bk-ml {border-left:3px solid #1e1e1e;min-width:46px;text-align:right;"
            "font-variant-numeric:tabular-nums;font-weight:600;font-size:0.82rem;white-space:nowrap}"
        ".bk-tot {min-width:72px;text-align:left;font-size:0.74rem;color:#555;padding-left:4px}"
        ".ou-pr {font-size:0.72rem}"
        ".ov .ou-pr {color:#5a8fc4}"
        ".un .ou-pr {color:#b06060}"

        # Badges
        ".badge {display:inline-block;width:15px;height:15px;border-radius:2px;"
            "font-size:0.55rem;font-weight:700;text-align:center;line-height:15px;"
            "margin-right:5px;vertical-align:middle}"
        ".ab {background:#1b3556;color:#6fa8dc}"
        ".hb {background:#1a3320;color:#6dbd6d}"
    )

    base_th = (
        "<th class='la'>Team</th>"
        "<th class='la'>Starting Pitcher</th>"
        "<th>Proj Runs</th>"
        "<th>F5 Total</th>"
        "<th>Win %</th>"
        "<th>Model FV</th>"
    )

    html_table = (
        "<!DOCTYPE html><html><head>"
        f"<style>{css}</style>"
        "</head><body>"
        f"<table class='t'><thead><tr>{base_th}{book_headers}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
        "</body></html>"
    )

    import streamlit.components.v1 as components
    components.html(html_table, height=max(len(today) * 68 + 50, 400), scrolling=False)

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
