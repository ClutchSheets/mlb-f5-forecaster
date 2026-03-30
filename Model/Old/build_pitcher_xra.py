"""
build_pitcher_xra.py
====================
Rebuilds f5_ra_by_pitcher.csv using exponential decay weighting
so recent starts count more than old ones.

Formula:
    weight[i] = DECAY ^ i   (i=0 is most recent, i=1 is second most recent...)
    xRA = sum(weight[i] * f5_allowed[i]) / sum(weight[i])

Blending:
    - 8+ starts:  100% recency-weighted (exponential decay over all starts)
    - 3-7 starts: blend of recency-weighted recent + career mean for stability
    - < 3 starts: excluded (not enough data)

Run this daily BEFORE f5_trackerv2.py so the xRA file is always fresh.

Usage:
    python build_pitcher_xra.py
    python build_pitcher_xra.py --decay 0.85   # custom decay rate
    python build_pitcher_xra.py --show-changes  # print biggest movers
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"D:\Desktop\F5 Baseball Forecaster\Model")
OUTPUT_DIR = Path(r"D:\Desktop\F5 Baseball Forecaster\Model")

RA_TRACKER  = DATA_DIR / "f5_ra_tracker.csv"
OUTPUT_FILE = OUTPUT_DIR / "f5_ra_by_pitcher.csv"

DECAY        = 0.85    # per-start decay: 0.85^0=1.0, 0.85^1=0.85, 0.85^7=0.32
                       # raise toward 1.0 for slower decay, lower for faster
MIN_STARTS   = 3       # exclude pitchers with fewer than this many starts
BLEND_CUTOFF = 8       # starts needed to rely fully on recency weighting
CAREER_BLEND = 0.35    # weight given to career mean when starts < BLEND_CUTOFF

# ── CORE FUNCTION ──────────────────────────────────────────────────────────
def compute_xra(group: pd.DataFrame, decay: float = DECAY) -> dict:
    """
    Given all starts for one pitcher (sorted oldest→newest),
    compute exponentially decayed xRA.
    """
    # Sort newest first for decay application
    runs = group.sort_values("date", ascending=False)["f5_allowed"].values
    n    = len(runs)

    # Exponential weights: most recent = weight 1.0, oldest = DECAY^(n-1)
    weights = np.array([DECAY ** i for i in range(n)])
    xra_recency = np.average(runs, weights=weights)

    # Career flat average for blending
    xra_career = runs.mean()

    # Blend if not enough starts for full confidence
    if n >= BLEND_CUTOFF:
        xra_final = xra_recency
    else:
        # Linear blend: more career weight when fewer starts
        career_w = CAREER_BLEND * (1 - (n - MIN_STARTS) / (BLEND_CUTOFF - MIN_STARTS))
        career_w = max(0.0, min(CAREER_BLEND, career_w))
        recency_w = 1.0 - career_w
        xra_final = recency_w * xra_recency + career_w * xra_career

    return {
        "f5_ra":         round(xra_final, 4),
        "f5_ra_recency": round(xra_recency, 4),
        "f5_ra_career":  round(xra_career, 4),
        "starts":        n,
        "last_start":    group["date"].max().strftime("%Y-%m-%d"),
    }

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decay", type=float, default=DECAY,
                        help=f"Decay rate per start (default {DECAY})")
    parser.add_argument("--show-changes", action="store_true",
                        help="Print biggest movers vs current file")
    args = parser.parse_args()

    decay = args.decay

    print(f"Building pitcher xRA  |  decay={decay}  min_starts={MIN_STARTS}")
    print(f"Source: {RA_TRACKER}")

    # Load tracker
    if not RA_TRACKER.exists():
        raise FileNotFoundError(f"Missing: {RA_TRACKER}")

    df = pd.read_csv(RA_TRACKER)
    df.columns  = df.columns.str.strip()
    df["date"]  = pd.to_datetime(df["date"])
    df["pitcher"] = df["pitcher"].str.strip()

    # Filter to starters with minimum appearances
    counts = df.groupby("pitcher").size()
    valid  = counts[counts >= MIN_STARTS].index
    df     = df[df["pitcher"].isin(valid)]

    print(f"Pitchers with {MIN_STARTS}+ starts: {len(valid)}")

    # Compute xRA per pitcher
    results = []
    for pitcher, group in df.groupby("pitcher"):
        stats = compute_xra(group, decay)
        stats["name"] = pitcher
        results.append(stats)

    new_df = (pd.DataFrame(results)
                [["name", "f5_ra", "f5_ra_recency", "f5_ra_career",
                  "starts", "last_start"]]
                .sort_values("f5_ra")
                .reset_index(drop=True))

    # Show changes vs existing file
    if args.show_changes and OUTPUT_FILE.exists():
        old_df = pd.read_csv(OUTPUT_FILE)
        merged = new_df.merge(old_df[["name","f5_ra"]].rename(
                     columns={"f5_ra":"f5_ra_old"}), on="name", how="left")
        merged["delta"] = merged["f5_ra"] - merged["f5_ra_old"]
        movers = merged.dropna(subset=["f5_ra_old"]).reindex(
                     merged["delta"].abs().sort_values(ascending=False).index)
        print("\nTop 15 biggest movers (recency vs career):")
        print(f"{'Pitcher':<28} {'Old':>6} {'New':>6} {'Delta':>7}")
        print("-" * 52)
        for _, r in movers.head(15).iterrows():
            arrow = "↓" if r["delta"] < 0 else "↑"
            print(f"{r['name']:<28} {r['f5_ra_old']:>6.3f} "
                  f"{r['f5_ra']:>6.3f} {arrow}{abs(r['delta']):>5.3f}")

    # Save — keep only name/f5_ra/starts for compatibility with f5_trackerv2.py
    out = new_df[["name", "f5_ra", "starts"]].copy()
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"  Pitchers: {len(out)}")
    print(f"  xRA range: {out['f5_ra'].min():.3f} – {out['f5_ra'].max():.3f}")
    print(f"  xRA mean:  {out['f5_ra'].mean():.3f}")

    # Spot check key names
    print("\nSpot checks:")
    for name in ["Eduardo Rodriguez", "Tyler Glasnow", "Zack Wheeler",
                 "Spencer Strider", "Kyle Bradish", "Sean Burke"]:
        row = out[out["name"] == name]
        if len(row):
            r    = row.iloc[0]
            full = new_df[new_df["name"] == name].iloc[0]
            print(f"  {name:<25} xRA={r['f5_ra']:.3f}  "
                  f"(recency={full['f5_ra_recency']:.3f}  "
                  f"career={full['f5_ra_career']:.3f}  "
                  f"n={r['starts']}  last={full['last_start']})")
        else:
            print(f"  {name:<25} NOT FOUND")

if __name__ == "__main__":
    main()
