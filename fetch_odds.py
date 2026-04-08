"""
fetch_odds.py
=============
Fetches F5 ML + F5 totals from The Odds API for today's MLB slate
and writes daily_odds.json.

Uses /events endpoint to get today's game IDs, then calls
/events/{eventId}/odds per game for h2h_1st_5_innings + totals_1st_5_innings.

Credit cost: (num_games x 2) per day. Typically ~20 credits for a full slate.

Called by 2_push_forecasts.bat — not by the app.
"""

import os
import json
import requests
from datetime import date, timezone, datetime

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "d31971f20a2edb069b228e828826815a")

BOOKS = [
    "betonlineag",  # BetOnline
    "betmgm",       # BetMGM
    "betrivers",    # BetRivers
    "draftkings",   # DraftKings
    "fanduel",      # FanDuel
]

OUTPUT_FILE = "daily_odds.json"
BASE_URL    = "https://api.the-odds-api.com/v4/sports/baseball_mlb"


def get_events() -> list:
    """Get today's MLB event IDs."""
    r = requests.get(
        f"{BASE_URL}/events",
        params={"apiKey": ODDS_API_KEY, "regions": "us"},
        timeout=15,
    )
    r.raise_for_status()
    today = date.today().isoformat()
    events = [
        e for e in r.json()
        if e.get("commence_time", "")[:10] == today
    ]
    print(f"  Found {len(events)} games today")
    return events


def get_event_odds(event_id: str, away: str, home: str, market: str) -> dict:
    """Fetch odds for a single event and market."""
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


def build_odds_dict(events: list) -> dict:
    games = {}
    total_calls = 0

    for event in events:
        event_id = event["id"]
        away     = event["away_team"]
        home     = event["home_team"]
        key      = f"{away}@{home}"

        games[key] = {"away": away, "home": home, "ml": {}, "total": {}}

        # ── F5 Moneyline ──
        try:
            data = get_event_odds(event_id, away, home, "h2h_1st_5_innings")
            total_calls += 1
            for bm in data.get("bookmakers", []):
                bk = bm["key"]
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "h2h_1st_5_innings":
                        continue
                    entry = {}
                    for o in mkt["outcomes"]:
                        side = "away" if o["name"] == away else "home"
                        entry[side] = int(o["price"])
                    games[key]["ml"][bk] = entry
        except Exception as e:
            print(f"  [WARN] ML failed for {away}@{home}: {e}")

        # ── F5 Totals ──
        try:
            data = get_event_odds(event_id, away, home, "totals_1st_5_innings")
            total_calls += 1
            for bm in data.get("bookmakers", []):
                bk = bm["key"]
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "totals_1st_5_innings":
                        continue
                    entry = {}
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            entry["line"]  = o.get("point")
                            entry["over"]  = int(o["price"])
                        else:
                            entry["under"] = int(o["price"])
                    games[key]["total"][bk] = entry
        except Exception as e:
            print(f"  [WARN] Totals failed for {away}@{home}: {e}")

        print(f"  [OK] {away} @ {home}")

    print(f"  Total API calls: {total_calls}")
    return games


def main():
    if ODDS_API_KEY == "YOUR_KEY_HERE":
        print("[ERROR] Open fetch_odds.py and replace YOUR_KEY_HERE with your key.")
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

    payload = {
        "date":  str(date.today()),
        "games": odds,
        "books": BOOKS,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUTPUT_FILE} — {len(odds)} games")


if __name__ == "__main__":
    main()
