"""
fetch_odds.py — ClutchSheets F5
Fetches per event:
  - F5 ML + F5 totals from US soft books (BOL, FD, DK, MGM, BR)
  - Pinnacle F5 ML (via spreads@0) + F5 totals as sharp reference
  
Pinnacle stored separately under games[key]["pinnacle"].
~3 credits per game (~30-45/day for full slate).
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
    try:
        dt  = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
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
    today  = date.today().isoformat()
    events = [e for e in r.json() if e.get("commence_time","")[:10] == today]
    print(f"  Found {len(events)} games today")
    return events


def get_event_odds(event_id, market, regions="us", bookmakers=None):
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    regions,
        "markets":    market,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    r = requests.get(
        f"{BASE_URL}/events/{event_id}/odds",
        params=params,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def build_odds_dict(events):
    games       = {}
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
            "pinnacle": {
                "ml":    {},   # {"away": int, "home": int}
                "total": {},   # {"line": float, "over": int, "under": int}
            },
        }

        # ── Soft book F5 ML ────────────────────────────────────
        try:
            data = get_event_odds(eid, "h2h_1st_5_innings",
                                  regions="us", bookmakers=BOOKS)
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

        # ── Soft book F5 Totals ────────────────────────────────
        try:
            data = get_event_odds(eid, "totals_1st_5_innings",
                                  regions="us", bookmakers=BOOKS)
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

        # ── Pinnacle F5 ML via spreads@0 (EU) ─────────────────
        # Pinnacle posts F5 as spreads_1st_5_innings at point=0
        # which is mathematically identical to a moneyline.
        try:
            data = get_event_odds(eid, "spreads_1st_5_innings",
                                  regions="eu", bookmakers=["pinnacle"])
            total_calls += 1
            for bm in data.get("bookmakers", []):
                if bm["key"] != "pinnacle": continue
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "spreads_1st_5_innings": continue
                    for o in mkt["outcomes"]:
                        # Only use handicap=0 outcomes
                        if o.get("point", 999) != 0: continue
                        side = "away" if o["name"] == away else "home"
                        games[key]["pinnacle"]["ml"][side] = int(o["price"])
            if games[key]["pinnacle"]["ml"]:
                print(f"  [PIN ML] {away}@{home}: {games[key]['pinnacle']['ml']}")
        except Exception as e:
            print(f"  [WARN] Pinnacle ML {away}@{home}: {e}")

        # ── Pinnacle F5 Totals (EU) ────────────────────────────
        try:
            data = get_event_odds(eid, "totals_1st_5_innings",
                                  regions="eu", bookmakers=["pinnacle"])
            total_calls += 1
            for bm in data.get("bookmakers", []):
                if bm["key"] != "pinnacle": continue
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "totals_1st_5_innings": continue
                    for o in mkt["outcomes"]:
                        if o["name"] == "Over":
                            games[key]["pinnacle"]["total"]["line"]  = o.get("point")
                            games[key]["pinnacle"]["total"]["over"]  = int(o["price"])
                        else:
                            games[key]["pinnacle"]["total"]["under"] = int(o["price"])
            if games[key]["pinnacle"]["total"]:
                print(f"  [PIN TOT] {away}@{home}: {games[key]['pinnacle']['total']}")
        except Exception as e:
            print(f"  [WARN] Pinnacle Tot {away}@{home}: {e}")

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

    fetched_at = (datetime.now(timezone.utc) - timedelta(hours=4)).strftime("%Y-%m-%d %I:%M %p EDT")
    payload = {
        "date":       str(date.today()),
        "fetched_at": fetched_at,
        "games":      odds,
        "books":      BOOKS,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUTPUT_FILE} — {len(odds)} games")


if __name__ == "__main__":
    main()
