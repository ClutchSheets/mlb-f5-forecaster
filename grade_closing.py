import os
import math
import time
import sqlite3
import requests
from datetime import datetime, timedelta, timezone

DB_FILE = "mlb_f5_forecasts.db"
API_KEY = os.environ.get("ODDS_API_KEY", "d31971f20a2edb069b228e828826815a")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
MARKET = "spreads_1st_5_innings"

TEAM_CODE_TO_NAME = {
    "PIT":"Pittsburgh Pirates","CHC":"Chicago Cubs","ATH":"Athletics","OAK":"Athletics",
    "NYM":"New York Mets","TEX":"Texas Rangers","LAD":"Los Angeles Dodgers","LAA":"Los Angeles Angels",
    "CIN":"Cincinnati Reds","WSN":"Washington Nationals","MIL":"Milwaukee Brewers",
    "ARI":"Arizona Diamondbacks","PHI":"Philadelphia Phillies","MIN":"Minnesota Twins","TOR":"Toronto Blue Jays",
    "COL":"Colorado Rockies","SDP":"San Diego Padres","MIA":"Miami Marlins","DET":"Detroit Tigers",
    "NYY":"New York Yankees","TBR":"Tampa Bay Rays","CHW":"Chicago White Sox","CWS":"Chicago White Sox",
    "KCR":"Kansas City Royals","SFG":"San Francisco Giants","BAL":"Baltimore Orioles","BOS":"Boston Red Sox",
    "STL":"St. Louis Cardinals","CLE":"Cleveland Guardians","ATL":"Atlanta Braves",
    "HOU":"Houston Astros","SEA":"Seattle Mariners"
}

def american_to_prob(o):
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def devig(p1, p2):
    s = p1 + p2
    return p1/s, p2/s

def clip(p):
    return max(1e-15, min(1-1e-15, p))

def logloss(p, y):
    p = clip(p)
    return -(y*math.log(p) + (1-y)*math.log(1-p))

def safe_get(url, params=None):
    delay = 1
    for _ in range(6):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r
        if r.status_code == 429:
            print(f"[429] sleeping {delay}s")
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
    raise Exception("Failed request")

def closing_timestamp(commence_time_str: str) -> str:
    """
    Return a timestamp 5 minutes before game start.
    The Odds API historical endpoint returns the most recent snapshot
    at or before the requested date — so commence_time - 5min reliably
    gives the true closing line rather than the opening snapshot.
    """
    ct = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
    close_dt = ct - timedelta(minutes=5)
    return close_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_cols(cur):
    cols = [c[1] for c in cur.execute("PRAGMA table_info(forecast_archive)")]
    def add(n, t):
        if n not in cols:
            cur.execute(f"ALTER TABLE forecast_archive ADD COLUMN {n} {t}")
    add("close_home_prob_devig", "REAL")
    add("close_logloss_devig", "REAL")
    add("model_vs_close_logloss_devig", "REAL")

def main():
    if API_KEY == "YOUR_API_KEY":
        raise Exception("Set ODDS_API_KEY")

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    ensure_cols(cur)

    rows = cur.execute("""
    SELECT rowid, date, away_team, home_team, winprob_home, home_win_f5
    FROM forecast_archive
    WHERE completed5=1 AND close_home_prob_devig IS NULL
    """).fetchall()

    print(f"Rows to backfill: {len(rows)}")

    for row in rows:
        rowid, date, away, home, p_model, y = row

        away_n = TEAM_CODE_TO_NAME.get(away)
        home_n = TEAM_CODE_TO_NAME.get(home)
        if not away_n or not home_n:
            print(f"  [SKIP] Unknown team code: {away} or {home}")
            continue

        # Find the event
        events = safe_get(
            f"{BASE_URL}/historical/sports/{SPORT}/events",
            {"apiKey": API_KEY, "date": date + "T12:00:00Z"}
        ).json()["data"]

        e = next((x for x in events if x["away_team"] == away_n and x["home_team"] == home_n), None)
        if not e:
            print(f"  [SKIP] Event not found: {away_n} @ {home_n} on {date}")
            continue

        # ── FIX: use closing timestamp (5 min before game start) ──────────
        close_ts = closing_timestamp(e["commence_time"])
        print(f"  [FETCH] {away}@{home} closing snapshot: {close_ts}")

        odds_resp = safe_get(
            f"{BASE_URL}/historical/sports/{SPORT}/events/{e['id']}/odds",
            {"apiKey": API_KEY, "date": close_ts,
             "bookmakers": "pinnacle", "markets": MARKET, "regions": "eu",
             "oddsFormat": "american"}
        ).json()["data"]

        # Guard against missing bookmaker data
        bookmakers = odds_resp.get("bookmakers", [])
        if not bookmakers or not bookmakers[0].get("markets"):
            print(f"  [SKIP] No Pinnacle odds found for {away}@{home}")
            continue

        outs = bookmakers[0]["markets"][0]["outcomes"]
        outs = [o for o in outs if o.get("point") == 0]
        if len(outs) != 2:
            print(f"  [SKIP] Expected 2 outcomes at point=0, got {len(outs)} for {away}@{home}")
            continue

        prices = {o["name"]: o["price"] for o in outs}
        if away_n not in prices or home_n not in prices:
            print(f"  [SKIP] Team name mismatch in outcomes for {away}@{home}")
            continue

        away_o = prices[away_n]
        home_o = prices[home_n]

        pa, ph = american_to_prob(away_o), american_to_prob(home_o)
        pa, ph = devig(pa, ph)

        ll_close = logloss(ph, y)
        ll_model = logloss(p_model, y)

        cur.execute("""
        UPDATE forecast_archive SET
            close_home_prob_devig=?,
            close_logloss_devig=?,
            model_vs_close_logloss_devig=?
        WHERE rowid=?
        """, (ph, ll_close, ll_model - ll_close, rowid))

        conn.commit()
        print(f"  [OK] {date} {away}@{home}  pin_close={away_o}/{home_o}  ph={ph:.6f}")

        time.sleep(1.1)

    conn.close()
    print("Backfill complete")

if __name__ == "__main__":
    main()
