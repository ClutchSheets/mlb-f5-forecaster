"""
fetch_odds.py — ClutchSheets F5
Fetches F5 ML + totals per event, stores commence_time for display.
2 credits per game (~20/day for full slate).
Called by 2_push_forecasts.bat only.
"""

import os, json, requests
from datetime import date, datetime, timezone, timedelta

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "d31971f20a2edb069b228e828826815a")

BOOKS = [
    "betonlineag",
    "fanduel",
    "draftkings",
    "betmgm",
    "betrivers",
]

OUTPUT_FILE = "daily_odds.json"
BASE_URL    = "https://api.the-odds-api.com/v4/sports/baseball_mlb"


def to_edt(iso_str):
    """Convert ISO UTC string to EDT time string e.g. '7:05 PM'."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        edt = dt - timedelta(hours=4)
        return edt.strftime("%-I:%M %p")
    except Exception:
        return ""


def get_events():
    r = requests.get(
        f"{BASE_URL}/events",
        params={"apiKey": ODDS_API_KEY, "regions": "us"},
        timeout=15,
    )
    r.raise_for_status()
    today = date.today().isoformat()
    events = [e for e in r.json() if e.get("commence_time","")[:10] == today]
    print(f"  Found {len(events)} games today")
    return events


def get_event_odds(event_id, market):
    r = requests.get(
        f"{BASE_URL}/events/{event_id}/odds",
        params={
            "apiKey":     ODDS_API_KEY,
            "regions":    "us",
            "markets":    market,
            "bookmakers": ",".join(BOOKS),
            "oddsFormat": "american",
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def build_odds_dict(events):
    games = {}
    total_calls = 0

    for event in events:
        eid   = event["id"]
        away  = event["away_team"]
        home  = event["home_team"]
        ctime = event.get("commence_time", "")
        key   = f"{away}@{home}"

        games[key] = {
            "away":         away,
            "home":         home,
            "commence_edt": to_edt(ctime),
            "ml":           {},
            "total":        {},
        }

        # F5 ML
        try:
            data = get_event_odds(eid, "h2h_1st_5_innings")
            total_calls += 1
            for bm in data.get("bookmakers", []):
                bk = bm["key"]
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "h2h_1st_5_innings": continue
                    entry = {}
                    for o in mkt["outcomes"]:
                        side = "away" if o["name"] == away else "home"
                        entry[side] = int(o["price"])
                    games[key]["ml"][bk] = entry
        except Exception as e:
            print(f"  [WARN] ML {away}@{home}: {e}")

        # F5 Total
        try:
            data = get_event_odds(eid, "totals_1st_5_innings")
            total_calls += 1
            for bm in data.get("bookmakers", []):
                bk = bm["key"]
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "totals_1st_5_innings": continue
                    entry = {}
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            entry["line"] = o.get("point")
                            entry["over"] = int(o["price"])
                        else:
                            entry["under"] = int(o["price"])
                    games[key]["total"][bk] = entry
        except Exception as e:
            print(f"  [WARN] Tot {away}@{home}: {e}")

        print(f"  [OK] {away} @ {home}  {to_edt(ctime)} EDT")

    print(f"  Total API calls: {total_calls}")
    return games


def main():
    if ODDS_API_KEY == "YOUR_KEY_HERE":
        print("[ERROR] Replace YOUR_KEY_HERE with your Odds API key.")
        return

    print(f"Fetching F5 odds for {date.today()} ...")

    try:
        events = get_events()
    except requests.HTTPError as e:
        print(f"[ERROR] Events fetch failed: {e}")
        return

    if not events:
        print("[SKIP] No games today.")
        return

    odds = build_odds_dict(events)

    from datetime import datetime, timezone, timedelta
    fetched_at = (datetime.now(timezone.utc) - timedelta(hours=4)).strftime("%Y-%m-%d %I:%M %p EDT")
    payload = {"date": str(date.today()), "fetched_at": fetched_at, "games": odds, "books": BOOKS}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUTPUT_FILE} — {len(odds)} games")


if __name__ == "__main__":
    main()
