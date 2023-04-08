# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% cell_id="2b4a45ae5af048bc90d9fa9a142f2e0f" deepnote_cell_type="code"
import datetime
import numpy as np
import pandas as pd
import pickle
import os

from datetime import timedelta

from paths_full import *

from features.holidays_feature.holidays_feature import add_ukrainian_holidays
from features.holidays_feature.holidays_feature import add_russian_holidays
from features.eclipses_feature.eclipses_feature import add_lunar_eclipses
from features.eclipses_feature.eclipses_feature import add_solar_eclipses

import duckdb

# %load_ext sql
# %config SqlMagic.autopandas = True
# %config SqlMagic.feedback = False
# %config SqlMagic.displaycon = False

# %sql duckdb:///:memory:

# %% cell_id="da035cf9b5734c08ac94916d26c71ca9" deepnote_cell_type="code"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# %% cell_id="ac6a0c9dc63949df851a22bff81b8535" deepnote_cell_type="code"
REPORTS_DATA_FILE = "./results/tfidf.csv"

OUTPUT_FOLDER = "results"
ISW_OUTPUT_DATA_FILE = "all_isw.csv"
WEATHER_EVENTS_OUTPUT_DATA_FILE = "all_hourly_weather_events.csv"

MODEL_FOLDER = "model"


# %% cell_id="593b7630e5f7480faf405e62afa70737" deepnote_cell_type="code"
def isNaN(num):
    return num != num


# %% [markdown] cell_id="03a410730b1c421ea9b1b77098d6611d" deepnote_cell_type="markdown"
# ## reading data

# %% cell_id="25b96939f84f4b7a9973fbbf25e0fa1a" deepnote_cell_type="code"
df_isw = pd.read_csv(f"{REPORTS_DATA_FILE}", sep=",")

# %% cell_id="6e60a9c9ceae45c8a33269edda7cf251" deepnote_cell_type="code"
df_isw.head(5)

# %% [markdown] cell_id="3db0ad4fc23c48b782e04ea66e73f028" deepnote_cell_type="markdown"
# ## preparing ISW reports

# %% [markdown] cell_id="960f89954d81433495c0f3f9c437a4b1" deepnote_cell_type="markdown"
# ## reading models

# %% cell_id="48ca4938d3ca4bca910faaf236295670" deepnote_cell_type="code"
df_isw["date_datetime"] = pd.to_datetime(df_isw["Date"])

# %% cell_id="3b3a59a75bb2437c86abb073be054278" deepnote_cell_type="code"
df_isw["date_tomorrow_datetime"] = df_isw["date_datetime"].apply(
    lambda x: x + datetime.timedelta(days=1)
)
df_isw["event_time"] = np.nan

# %% cell_id="52ddc5c50edf4c579e5868a0de49c450" deepnote_cell_type="code"
df_isw = df_isw.rename(columns={"date_datetime": "report_date"})
df_isw.to_csv(f"{OUTPUT_FOLDER}/{ISW_OUTPUT_DATA_FILE}", sep=";", index=False)

# %% cell_id="cfe8d26df3a34c3b8c600d45c8b714d2" deepnote_cell_type="code"
# Add holidays data to df_isw
add_ukrainian_holidays(df_isw, day_datetime_column='report_date', column_name='ukrainian_holiday')
add_russian_holidays(df_isw, day_datetime_column='report_date', column_name='russian_holiday')
# df_isw.loc[df_isw['ukrainian_holiday'] == 1]

# %% [markdown] cell_id="ced25f3d8b2c4d96b86451c7bff747a6" deepnote_cell_type="markdown"
# ## prepare events data

# %% cell_id="661f54f58e0a4e368b0960ea1f4960bd" deepnote_cell_type="code"
df_events = pd.read_csv(f"{ALARMS_DATA_FILE}", sep=";")

# %% cell_id="ce00bb16d2e94d128d28f352bde22cbb" deepnote_cell_type="code"
df_events_v2 = df_events.drop(["id", "region_id"], axis=1)
df_events_v2["event_time"] = np.nan

# %% cell_id="43771d041c58424eb692f73230f4b791" deepnote_cell_type="code"
df_events_v2.head(5)
df_events_v2.shape

# %% cell_id="616debc8773049ceb42e4d4bd662954b" deepnote_cell_type="code"
# df_events_v2["start_time"] = df_events_v2.apply(lambda x: x["start"] if not isNaN(x["start"]) else x["event_time"] , axis=1)
# df_events_v2["end_time"] = df_events_v2.apply(lambda x: x["end"] if not isNaN(x["end"]) else x["event_time"], axis=1)

# %% cell_id="613e468d21b7436dbe18a526bcf8d296" deepnote_cell_type="code"
df_events_v2["start_time"] = pd.to_datetime(df_events_v2["start"])
df_events_v2["end_time"] = pd.to_datetime(df_events_v2["end"])
df_events_v2["event_time"] = pd.to_datetime(df_events_v2["event_time"])

# %% cell_id="3797b89c253e4201969251924a6c753f" deepnote_cell_type="code"
df_events_v2["start_hour"] = df_events_v2["start_time"].dt.floor("H")
df_events_v2["end_hour"] = df_events_v2["end_time"].dt.ceil("H")
df_events_v2["event_hour"] = df_events_v2["event_time"].dt.round("H")

# %% cell_id="fddc3ee690664ca9ab4d0c8dce8a3138" deepnote_cell_type="code"
df_events_v2["start_hour"] = df_events_v2.apply(
    lambda x: x["start_hour"] if not isNaN(x["start_hour"]) else x["event_hour"], axis=1
)
df_events_v2["end_hour"] = df_events_v2.apply(
    lambda x: x["end_hour"] if not isNaN(x["end_hour"]) else x["event_hour"], axis=1
)

# %% cell_id="c4638f2955c74e41995e3ddaf348c408" deepnote_cell_type="code"
df_events_v2["day_date"] = df_events_v2["start_time"].dt.date

df_events_v2.head(10)

df_events_v2["start_hour_datetimeEpoch"] = df_events_v2["start_hour"].apply(
    lambda x: int(x.timestamp()) if not isNaN(x) else None
)
df_events_v2["end_hour_datetimeEpoch"] = df_events_v2["end_hour"].apply(
    lambda x: int(x.timestamp()) if not isNaN(x) else None
)

# df_events_v2.head(10)

# %% [markdown] cell_id="60915435755f423ba7455e694efcff97" deepnote_cell_type="markdown"
# ## prepare weather

# %% cell_id="128e051e53694b4f92b1d765f6e5eb4b" deepnote_cell_type="code"
df_weather = pd.read_csv(f"{WEATHER_DATA_FILE}")
df_weather["day_datetime"] = pd.to_datetime(df_weather["day_datetime"])

# %% cell_id="6ab1b03027504c218ed5e02b5944ee28" deepnote_cell_type="code"
df_weather.shape

# %% cell_id="ca2c2b5a13334a78b21504402d417830" deepnote_cell_type="code"
df_weather.head(15)

# %% cell_id="f10f1491c1b44ee1ab63f17929ffff38" deepnote_cell_type="code"
# exclude
weather_exclude = [
    "day_feelslikemax",
    "day_feelslikemin",
    "day_sunriseEpoch",
    "day_sunsetEpoch",
    "day_description",
    "city_latitude",
    "city_longitude",
    "city_address",
    "city_timezone",
    "city_tzoffset",
    "day_feelslike",
    "day_precipprob",
    "day_snow",
    "day_snowdepth",
    "day_windgust",
    "day_windspeed",
    "day_winddir",
    "day_pressure",
    "day_cloudcover",
    "day_visibility",
    "day_severerisk",
    "day_conditions",
    "day_icon",
    "day_source",
    "day_preciptype",
    "day_stations",
    "hour_icon",
    "hour_source",
    "hour_stations",
    "hour_feelslike",
]

# %% cell_id="02dffa78339b409eba1906019ae553fe" deepnote_cell_type="code"
df_weather_v2 = df_weather.drop(weather_exclude, axis=1)

# %% cell_id="bd136660563e43478b9d934a89d5da1b" deepnote_cell_type="code"
df_weather_v2["city"] = df_weather_v2["city_resolvedAddress"].apply(
    lambda x: x.split(",")[0]
)
df_weather_v2["city"] = df_weather_v2["city"].replace(
    "Хмельницька область", "Хмельницький"
)

# %% cell_id="7035ab608f4f440aacd6a30d55f981d8" deepnote_cell_type="code"
df_weather_v2.head(5)

# %% cell_id="2a2e76b0bf174bcb98622170e00d66c4" deepnote_cell_type="code"
df_weather_v2.shape

# %% [markdown] cell_id="12b18e5dfd784c70b4170075378f7080" deepnote_cell_type="markdown"
# ## merging data

# %% cell_id="601c4a8529864ae9a16c0630e4e3b64d" deepnote_cell_type="code"
REGIONS_DATA_FOLDER = "../external_data/additions"
REGIONS_DATA_FILE = "regions.csv"
df_regions = pd.read_csv(f"{REGIONS_DATA_FOLDER}/{REGIONS_DATA_FILE}")

# %% cell_id="204c1f5676cb438b87b5523b625886a3" deepnote_cell_type="code"
df_regions.head(5)

# %% cell_id="c07fbbbdcf584e8993cc7d4d4a8b8651" deepnote_cell_type="code"
df_weather_reg = pd.merge(
    df_weather_v2, df_regions, left_on="city", right_on="center_city_ua"
)

# %% cell_id="44a4b2b9da614135b11dcf0c31ff90cc" deepnote_cell_type="code"
df_weather_reg.head(10)

# %% cell_id="21a4956d1ab040ed947ed172556665b6" deepnote_cell_type="code"
df_weather_reg.shape

# %% cell_id="95f76614fa894bd5aa924679adfdc682" deepnote_cell_type="code"
df_weather_v2.shape

# %% [markdown] cell_id="553fbeced23c40a48a3504d00b1d70e9" deepnote_cell_type="markdown"
# ### Merging weather and events

# %% cell_id="87bc6f45d1e94e918ac17de085309d75" deepnote_cell_type="code"
# df_events_v2["start_hour_datetimeEpoch"] = df_events_v2['start_hour'].apply(lambda x: int(x.strftime('%s'))  if not isNaN(x) else 0)
# df_events_v2["end_hour_datetimeEpoch"] = df_events_v2['end_hour'].apply(lambda x: int(x.strftime('%s'))  if not isNaN(x) else 0)

# %% cell_id="40055894945e4707977c52ac7def16b6" deepnote_cell_type="code"
df_events_v2.dtypes

# %% cell_id="ca638e87e4ad4d4fb75a1833e5ae7f4c" deepnote_cell_type="code"
df_events_v2.shape

# %% cell_id="6d40bd1596004f1abef8ccf066f59e9d" deepnote_cell_type="code"
df_events_v2.head(10)

# %% cell_id="f563ac4a436740d1a1e69e78e4a4b3a2" deepnote_cell_type="code"
# df_events_v2_sample = df_events_v2.sample(10)
# df_events_v2_sample.shape

events_dict = df_events_v2.to_dict("records")
events_by_hour = []

# %% cell_id="82db729f94fc4f5687711f6d159908b0" deepnote_cell_type="code"
events_dict[0]

# %% cell_id="7aee3ce8dc714f4fa7e122c64cf8f47b" deepnote_cell_type="code"
for event in events_dict:
    for d in pd.date_range(start=event["start_hour"], end=event["end_hour"], freq="1H"):
        et = event.copy()
        et["hour_level_event_time"] = d
        events_by_hour.append(et)

# %% cell_id="8fa0f3a05e4d4d8ea474bf45e2db1f07" deepnote_cell_type="code"
df_events_v3 = pd.DataFrame.from_dict(events_by_hour)
df_events_v3["hour_level_event_datetimeEpoch"] = df_events_v3[
    "hour_level_event_time"
].apply(lambda x: int(x.timestamp()) if not isNaN(x) else None)

# %% cell_id="f9e6d8dcd0c24da882366d0919c71b95" deepnote_cell_type="code"
df_events_v3.shape

# %% cell_id="8e4a4854d0c1438c8f6ea96d590648cf" deepnote_cell_type="code"
df_events_v3.head(15)

# %% cell_id="ff28d9be8f154e029b442fee3319b359" deepnote_cell_type="code"
df_weather_reg.head(5)

# %% cell_id="e6dbcef495b24bcfa953dc2bba6b35cb" deepnote_cell_type="code"
df_weather_reg.shape

# %% cell_id="1b4bf461c6b140abbaca196778092760" deepnote_cell_type="code"
df_events_v3.head(10)
df_weather_reg.head(10)

# %% cell_id="ec545fd52e5c40a7a58a7293301fa52b" deepnote_cell_type="code"
df_events_v4 = df_events_v3.copy().add_prefix("event_")
df_events_v4.head(10)

# %% cell_id="c196ef5769f14944b471c1e9c8182dbd" deepnote_cell_type="code"
df_weather_v4 = df_weather_reg.merge(
    df_events_v4,
    how="left",
    left_on=["region_alt", "hour_datetimeEpoch"],
    right_on=["event_region_title", "event_hour_level_event_datetimeEpoch"],
)

# %% cell_id="db4146abf636459a9947e6ae334ec013" deepnote_cell_type="code"
# Alarm data
print(df_weather_v4.loc[~ isNaN(df_weather_v4['event_start'])].shape)
print(df_weather_v4.loc[isNaN(df_weather_v4['event_start'])].shape)
df_weather_v4['is_alarm'] = df_weather_v4.apply(lambda x: 0 if isNaN(x['event_start']) else 1, axis=1)
no_alarms = df_weather_v4.loc[df_weather_v4['is_alarm'] == 0].size
alarms = df_weather_v4.loc[df_weather_v4['is_alarm'] == 1].size
print(f"Alarm chane: {alarms / df_weather_v4.size}")
print(f"No alarm: {no_alarms / df_weather_v4.size}")

# %% cell_id="d432c747179c4a37bc49d000336fa478" deepnote_cell_type="code"
add_lunar_eclipses(df_weather_v4, day_datetime_column="day_datetime")
add_solar_eclipses(df_weather_v4, day_datetime_column="day_datetime")

# %% cell_id="7f28cf0beccc4ea2b0c7929652981950" deepnote_cell_type="code"
# Alarm data
print(df_weather_v4.loc[~ isNaN(df_weather_v4['event_start'])].shape)
print(df_weather_v4.loc[isNaN(df_weather_v4['event_start'])].shape)
df_weather_v4['is_alarm'] = df_weather_v4.apply(lambda x: 0 if isNaN(x['event_start']) else 1, axis=1)
no_alarms = df_weather_v4.loc[df_weather_v4['is_alarm'] == 0].size
alarms = df_weather_v4.loc[df_weather_v4['is_alarm'] == 1].size
print(f"Alarm chane: {alarms / df_weather_v4.size}")
print(f"No alarm: {no_alarms / df_weather_v4.size}")
# df_weather_v4.sample(5)

# %% cell_id="599b7a5d8544490788c58e2aa783df62" deepnote_cell_type="code"
# Merge isw data to df_weather_v4
df_weather_v5 = pd.merge(
    df_weather_v4,
    df_isw[
        [
            "Keywords",
            "report_date",
            "date_tomorrow_datetime",
            "ukrainian_holiday",
            "russian_holiday",
        ]
    ],
    left_on="day_datetime",
    right_on="report_date",
)

print(df_weather_v5.shape)

df_weather_v6 = None
df_weather_v5[df_weather_v5["event_start_time"] > pd.to_datetime(0)].head(5)

# %% cell_id="dd3ec300bbc44b72ae1ef018237f8e50" deepnote_cell_type="code"
# df_weather_v5.to_csv(f"./results/df_v5.csv")

# %% [markdown] cell_id="6a2458f26b8140e589b3d2bb3cadd515" deepnote_cell_type="markdown"
# ## Feature engineering

# %% [markdown] cell_id="ad9519e8c60a4645ad9d5159748f1dd6" deepnote_cell_type="markdown"
# ### Number of alarms for this region during the last 24 hours

# %% [markdown] cell_id="698ce5e96db24b2fa7a53e4f40fe617d" deepnote_cell_type="markdown"
# Use DuckDB for analytics as running with Pandas and Python taking too long.

# %% cell_id="3c0032ffd14144caab2b1bed8db9ab1b" deepnote_cell_type="code" language="sql"
# df_weather_v6 << select df.*, coalesce(alarm_count.events_last_24_hrs, 0) as events_last_24_hrs
# from df_weather_v5 df
#          left join (select out.region_id,
#                             out.hour_datetimeEpoch,
#                             count(inn.event_start_time) as events_last_24_hrs
#                      from df_weather_v5 out
#                               left join df_weather_v5 inn
#                                          on out.region_id = inn.region_id
#                      where
#                        inn.event_start_time::timestamp
#                         between
#                         (epoch_ms(out.hour_datetimeEpoch::long * 1000) - '24 HOURS'::interval)
#                             and epoch_ms(out.hour_datetimeEpoch::long * 1000)
#                      group by out.region_id, out.hour_datetimeEpoch) as alarm_count
#         on df.region_id = alarm_count.region_id and df.hour_datetimeEpoch = alarm_count.hour_datetimeEpoch;

# %% cell_id="f2996c9e0cf84c1595b02ae9db365503" deepnote_cell_type="code"
print(df_weather_v6.shape)

df_weather_v6[['city_resolvedAddress', 'region_id', 'hour_datetimeEpoch', 'events_last_24_hrs']].tail(5)

# %% cell_id="6818c8ea9fdf4c40990e193cc1605126" deepnote_cell_type="code"
df_weather_v6.tail(1)

# %% cell_id="7a80f3a8324f4f759fe717196aa8b9e3" deepnote_cell_type="code"
# Add day of week name
df_weather_v6["day_of_week"] = df_weather_v6["day_datetime"].apply(
    lambda date: pd.to_datetime(date).day_name()
)

df_weather_v6[["day_datetime", "day_of_week"]].head(10)

# %% [markdown] cell_id="8e65380583fa44668f1d4fec361333cb" deepnote_cell_type="markdown"
# ### Handle categorical data

# %% cell_id="ae8168b0e88f49d5b5c356e147aa16d9" deepnote_cell_type="code"
# Encode days of week into one hot encoding for linear regression
df_weather_v6 = pd.get_dummies(
    df_weather_v6, columns=["day_of_week"], prefix=["day_of_week"]
)

df_weather_v6[
    [
        "day_datetime",
        "day_of_week_Monday",
        "day_of_week_Tuesday",
        "day_of_week_Wednesday",
        "day_of_week_Thursday",
        "day_of_week_Friday",
        "day_of_week_Saturday",
        "day_of_week_Sunday",
    ]
].head(10)

# %% cell_id="69bd2164749343b29989f247551186a3" deepnote_cell_type="code"
df_weather_v6["hour_conditions"] = pd.Categorical(df_weather_v6['hour_conditions'])
df_weather_v6["hour_conditions_code"] = df_weather_v6["hour_conditions"].cat.codes
df_weather_v6[["hour_conditions", "hour_conditions_code"]].head(5)

# %% cell_id="0bf3440e4f6d408cad0f2db500e44350" deepnote_cell_type="code"
df_weather_v7 = None

# %% cell_id="5e36606968b14eda94a77afed0a068ea" deepnote_cell_type="code" language="sql"
# df_weather_v7 << select df.*, coalesce(alarmed_count.alarmed_regions_count, 0) as alarmed_regions_count
# from df_weather_v6 df
#          left join (select out.day_datetimeEpoch,
#                                             out.hour_datetimeEpoch,
#                                             count(*) as alarmed_regions_count
#                                     from df_weather_v6 out left join df_weather_v6 inn
#                                          on out.day_datetimeEpoch = inn.day_datetimeEpoch and out.hour_datetimeEpoch = inn.hour_datetimeEpoch
#                                     where inn.is_alarm = 1 and inn.hour_datetimeEpoch
#                                                                between out.event_start_hour_datetimeEpoch and out.event_start_hour_datetimeEpoch
#                                     group by out.day_datetimeEpoch, out.hour_datetimeEpoch) as alarmed_count
#         on df.day_datetimeEpoch = alarmed_count.day_datetimeEpoch and df.hour_datetimeEpoch = alarmed_count.hour_datetimeEpoch;

# %% [markdown] cell_id="69c38d2760aa4198b6a49ea956f06e52" deepnote_cell_type="markdown"
# ### Save final merged dataframe

# %% cell_id="8b906aa46c70419b8c5a2aeb4be211e0" deepnote_cell_type="code"
df_weather_v7.to_csv(
    f"{OUTPUT_FOLDER}/df_weather_v7.csv", sep=";", index=False
)