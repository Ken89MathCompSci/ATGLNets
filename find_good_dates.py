"""
Scans the UKDALE HDF5 file and finds dates where all appliances
in a given house have >= MIN_SAMPLES resampled samples (at 6s intervals).

Run: python find_good_dates.py
"""

import numpy as np
import pandas as pd
from data_loader import (
    H5_PATH, TARGET_APPLIANCES, HOUSE_METER_MAP, RESAMPLE_FREQ,
    read_meter, TARGET_HOUSES
)

MIN_SAMPLES  = 14000   # ~97% of a full day at 6s -> 14,400
SCAN_HOUSES  = [1, 2, 5]


def daily_counts(series, start_date, end_date):
    """
    For each calendar date in [start_date, end_date], count how many
    resampled samples exist after slicing to that day.
    Returns a dict {date_str: count}.
    """
    counts = {}
    dates = pd.date_range(start=start_date, end=end_date, freq='D', tz='UTC')
    for day in dates:
        day_start = day
        day_end   = day + pd.Timedelta(hours=23, minutes=59, seconds=54)
        sliced    = series.loc[day_start:day_end]
        resampled = sliced.resample(RESAMPLE_FREQ).mean().ffill().fillna(0)
        counts[str(day.date())] = len(resampled)
    return counts


def scan_house(h5_path, building, scan_start='2013-01-01', scan_end='2014-12-31'):
    print(f"\n{'='*60}")
    print(f"House {building}  —  scanning {scan_start} to {scan_end}")
    print(f"{'='*60}")

    meter_map = HOUSE_METER_MAP.get(building, {})

    # Load mains
    try:
        mains = read_meter(h5_path, building, meter=1)
    except Exception as e:
        print(f"  Cannot read mains: {e}")
        return

    print(f"  Mains range: {mains.index[0]} -> {mains.index[-1]}")

    # Build per-appliance daily count dicts
    app_counts = {}   # appliance_name -> {date_str: count}
    for appliance in TARGET_APPLIANCES:
        meter = meter_map.get(appliance)
        if meter is None:
            continue
        try:
            series = read_meter(h5_path, building, meter)
        except KeyError:
            continue
        print(f"  Scanning {appliance} (meter {meter})...", end=' ', flush=True)
        app_counts[appliance] = daily_counts(series, scan_start, scan_end)
        good = sum(1 for v in app_counts[appliance].values() if v >= MIN_SAMPLES)
        print(f"{good} good days out of {len(app_counts[appliance])}")

    if not app_counts:
        print("  No appliances loaded.")
        return

    # Find dates where ALL present appliances meet the threshold
    all_dates = sorted(next(iter(app_counts.values())).keys())
    good_dates = []
    for date in all_dates:
        if all(app_counts[app].get(date, 0) >= MIN_SAMPLES for app in app_counts):
            good_dates.append(date)

    print(f"\n  Appliances checked: {list(app_counts.keys())}")
    print(f"  Dates where ALL appliances have >= {MIN_SAMPLES} samples:")
    if not good_dates:
        print("    (none found in this range)")
    else:
        # Group into contiguous runs for readability
        from itertools import groupby
        from datetime import datetime, timedelta

        runs = []
        run_start = None
        prev = None
        for d in good_dates:
            dt = datetime.strptime(d, '%Y-%m-%d')
            if prev is None or (dt - prev).days > 1:
                run_start = dt
            runs.append((run_start, dt))
            prev = dt

        # Deduplicate to just show run extents
        seen_starts = set()
        for rs, re in runs:
            key = rs.strftime('%Y-%m-%d')
            if key not in seen_starts:
                seen_starts.add(key)
        # Print unique consecutive runs
        unique_runs = []
        for d in good_dates:
            dt = datetime.strptime(d, '%Y-%m-%d')
            if not unique_runs or (dt - datetime.strptime(unique_runs[-1][1], '%Y-%m-%d')).days > 1:
                unique_runs.append([d, d])
            else:
                unique_runs[-1][1] = d

        for rs, re in unique_runs:
            n = (datetime.strptime(re, '%Y-%m-%d') - datetime.strptime(rs, '%Y-%m-%d')).days + 1
            print(f"    {rs}  ->  {re}  ({n} consecutive days)")

        print(f"\n  Total good dates: {len(good_dates)}")
        print(f"  Suggested splits (non-overlapping):")

        # Suggest: take first third, middle third, last third of good_dates
        n = len(good_dates)
        if n >= 3:
            train_date = good_dates[0]
            val_date   = good_dates[n // 2]
            test_date  = good_dates[-1]
            # Ensure they're far apart (at least 7 days)
            from datetime import datetime
            candidates = good_dates
            # train = first, val = ~middle but >= 7 days from train,
            # test  = last but >= 7 days from val
            train_dt = datetime.strptime(train_date, '%Y-%m-%d')
            val_candidates = [d for d in candidates
                              if (datetime.strptime(d, '%Y-%m-%d') - train_dt).days >= 7]
            if val_candidates:
                val_date   = val_candidates[len(val_candidates) // 2]
                val_dt     = datetime.strptime(val_date, '%Y-%m-%d')
                test_candidates = [d for d in candidates
                                   if (datetime.strptime(d, '%Y-%m-%d') - val_dt).days >= 7]
                test_date = test_candidates[-1] if test_candidates else good_dates[-1]

            print(f"    train: '{train_date} 00:00:00'  ->  '{train_date} 23:59:54'")
            print(f"    val:   '{val_date} 00:00:00'  ->  '{val_date} 23:59:54'")
            print(f"    test:  '{test_date} 00:00:00'  ->  '{test_date} 23:59:54'")


if __name__ == "__main__":
    for house in SCAN_HOUSES:
        scan_house(H5_PATH, house, scan_start='2013-01-01', scan_end='2014-12-31')
