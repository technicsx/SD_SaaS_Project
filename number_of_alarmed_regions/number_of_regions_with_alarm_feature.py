import os

import pandas as pd

import numpy as np

from dotenv import load_dotenv

load_dotenv("../.path_env")

EVENTS_DATA_FILE = "../" + os.getenv("ALARMS_DATA_FILE")
df_events = pd.read_csv(f"{EVENTS_DATA_FILE}", sep=";")

df_events_v2 = df_events.drop(["id", "region_id"], axis=1)

df_events_v2["event_time"] = np.nan
df_events_v2["start_time"] = pd.to_datetime(df_events_v2["start"])
df_events_v2["end_time"] = pd.to_datetime(df_events_v2["end"])
df_events_v2["event_time"] = pd.to_datetime(df_events_v2["event_time"])
df_events_v2["start_hour"] = df_events_v2["start_time"].dt.floor("H")
df_events_v2["end_hour"] = df_events_v2["end_time"].dt.ceil("H")
df_events_v2["event_hour"] = df_events_v2["event_time"].dt.round("H")
def isNaN(num):
    return num != num

df_events_v2["start_hour"] = df_events_v2.apply(
    lambda x: x["start_hour"] if not isNaN(x["start_hour"]) else x["event_hour"], axis=1
)
df_events_v2["end_hour"] = df_events_v2.apply(
    lambda x: x["end_hour"] if not isNaN(x["end_hour"]) else x["event_hour"], axis=1
)
df_events_v2["day_date"] = df_events_v2["start_time"].dt.date

df_events_v2["start_hour_datetimeEpoch"] = df_events_v2["start_hour"].apply(
    lambda x: int(x.timestamp()) if not isNaN(x) else None
)
df_events_v2["end_hour_datetimeEpoch"] = df_events_v2["end_hour"].apply(
    lambda x: int(x.timestamp()) if not isNaN(x) else None
)

def add_regions_number_with_alarm(dataset, column_name, df_events):
    dataset[column_name] = dataset.apply(lambda x: count_events_in_hour(x['day_datetime'], x['hour_datetime'], df_events))
def count_events_in_hour(date, hour_epoch, df_events):
    regions = set()
    df = df_events.to_dict("records")
    for row in df:

        start_hour = row["start_hour_datetimeEpoch"]
        end_hour = row["end_hour_datetimeEpoch"]

        event_date = row["day_date"].strftime('%Y-%m-%d')

        if event_date == date and start_hour <= hour_epoch < end_hour:
            print(row["region_title"])
            regions.add(row['region_title'])
    return len(regions)


if __name__ == '__main__':
    print(count_events_in_hour('2022-02-27', 1645894800, df_events_v2))
