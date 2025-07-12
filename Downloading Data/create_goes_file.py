from sunpy.net import Fido, attrs as a
from datetime import datetime
import pandas as pd

START_YEAR = 2011    # SDO/HMI started
END_YEAR = 2024      # last year to check
OUTPUT_FILE = 'goes_flares.csv'
MIN_CLASS = None     # for example: "M1.0" if you want M-class and above

all_flares = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Fetching GOES flares for {year}...")
    t_start = datetime(year, 1, 1)
    t_end = datetime(year + 1, 1, 1)

    search_attrs = [
        a.Time(t_start, t_end),
        a.hek.EventType("FL"),
        a.hek.OBS.Observatory == "GOES"
    ]

    if MIN_CLASS:
        search_attrs.append(a.hek.FL.GOESCls >= MIN_CLASS)

    results = Fido.search(*search_attrs)

    if not results:
        print(f"No flares found for {year}.")
        continue

    event_table = results['hek'][
        "event_starttime", "event_peaktime", "event_endtime",
        "fl_goescls", "ar_noaanum"
    ]

    event_df = event_table.to_pandas().rename(columns={
        'event_starttime': 'start_time',
        'event_peaktime': 'peak_time',
        'event_endtime': 'end_time',
        'fl_goescls': 'goes_class',
        'ar_noaanum': 'noaa_active_region'
    })

    # Remove entries with no NOAA AR assigned (0)
    event_df = event_df[event_df['noaa_active_region'] != 0]

    all_flares.append(event_df)

# Concatenate all years
if all_flares:
    goes_flares_df = pd.concat(all_flares).reset_index(drop=True)
    # Optional: Remove empty class entries
    goes_flares_df = goes_flares_df[goes_flares_df['goes_class'].notna()]
    # Save
    goes_flares_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(goes_flares_df)} flares to {OUTPUT_FILE}")
else:
    print("No flares found in the specified period.")
