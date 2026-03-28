import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import altair as alt

DB_FILE = "mlb_f5_forecasts.db"

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
    import os

    FORECAST_FILE = "daily_forecast.csv"

    @st.cache_data(ttl=300)
    def load_forecast_csv():
        if not os.path.exists(FORECAST_FILE):
            return pd.DataFrame()
        fc = pd.read_csv(FORECAST_FILE)
        fc["date"] = fc["date"].astype(str).str.strip()
        return fc

    csv_df = load_forecast_csv()

    if csv_df.empty:
        st.warning(f"No forecast file found. Drop daily_forecast.csv next to app.py.")
        st.stop()

    forecast_date = csv_df["date"].max()
    st.title(f"MLB F5 Forecasts — {forecast_date}")
    st.caption("Projections from today's model run. Results scored overnight.")

    today = csv_df[csv_df["date"] == forecast_date].copy()

    if "away_sp" not in today.columns:
        today["away_sp"] = ""
    if "home_sp" not in today.columns:
        today["home_sp"] = ""

    today["lambda_away"] = to_num(today["lambda_away"]).round(2)
    today["lambda_home"] = to_num(today["lambda_home"]).round(2)
    today["total_proj"] = to_num(today["total_proj"]).round(2)
    today["winprob_home"] = (to_num(today["winprob_home"]) * 100).round(1)

    today = today.rename(columns={
        "away_team": "Away",
        "away_sp": "Away SP",
        "home_team": "Home",
        "home_sp": "Home SP",
        "lambda_away": "Proj F5 Runs (Away)",
        "lambda_home": "Proj F5 Runs (Home)",
        "total_proj": "Proj F5 Total",
        "winprob_home": "Home Win %"
    })

    cols = [
        "Away", "Away SP",
        "Home", "Home SP",
        "Proj F5 Runs (Away)",
        "Proj F5 Runs (Home)",
        "Proj F5 Total",
        "Home Win %"
    ]

    st.dataframe(
        today[cols].sort_values("Home Win %", ascending=False),
        use_container_width=True,
        hide_index=True
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
        dv = dv.sort_values("created_at").copy()
        dv["rolling_mae_7"] = to_num(dv["mae_total"]).rolling(7).mean()
        dv["rolling_brier_7"] = to_num(dv["brier"]).rolling(7).mean()
        dv["rolling_logloss_7"] = to_num(dv["logloss"]).rolling(7).mean()

        st.subheader("Rolling 7-Game Metrics")
        st.line_chart(
            dv.set_index("created_at")[["rolling_mae_7", "rolling_brier_7", "rolling_logloss_7"]]
        )
    else:
        st.info("Rolling metrics will appear once you have timestamps and more rows.")

# -----------------------------
# CALIBRATION
# -----------------------------
elif page == "Calibration":
    st.title("Win Probability Calibration")

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
