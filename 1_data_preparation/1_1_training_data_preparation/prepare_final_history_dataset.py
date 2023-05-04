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

# %%

# %%
# !pip install jupysql
# !pip install duckdb
# !pip install duckdb-engine

# %% cell_id="2b4a45ae5af048bc90d9fa9a142f2e0f" deepnote_cell_type="code"
import datetime
import numpy as np
import pandas as pd
import pickle
import os

from datetime import timedelta


from src.features.holidays_feature import add_ukrainian_holidays
from src.features.eclipses_feature import add_lunar_eclipses

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
REPORTS_DATA_FILE = "./results/tfidf-history.csv"

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
df_isw = pd.read_csv("./results/tfidf-history.csv", sep=",")
df_isw.head(5)

# %% [markdown] cell_id="3db0ad4fc23c48b782e04ea66e73f028" deepnote_cell_type="markdown"
# ## preparing ISW reports

# %% [markdown] cell_id="960f89954d81433495c0f3f9c437a4b1" deepnote_cell_type="markdown"
# ISW reports intial preprocessing

# %% cell_id="48ca4938d3ca4bca910faaf236295670" deepnote_cell_type="code"
df_isw["report_date"] = pd.to_datetime(df_isw["Date"])
df_isw.drop(columns=["Date", "Name"], inplace=True)
df_isw.head(1)

# %% cell_id="3b3a59a75bb2437c86abb073be054278" deepnote_cell_type="code"
df_isw["date_tomorrow_datetime"] = df_isw["report_date"].apply(
    lambda x: x + datetime.timedelta(days=1)
)
# convert date_tomorrow_datetime to epoch in seconds
df_isw["date_tomorrow_epoch"] = df_isw["date_tomorrow_datetime"].apply(
    lambda x: int(x.timestamp())
)
# minus 2 hours to get the same day
df_isw["date_tomorrow_epoch"] = df_isw["date_tomorrow_epoch"] - 7200

df_isw["event_time"] = np.nan
df_isw.head(1)

# %%
df_isw_tmp = df_isw.copy()
df_isw_tmp["Keywords"] = df_isw_tmp["Keywords"].apply(lambda x: dict(eval(x)))

df_isw_v2 = pd.DataFrame(df_isw_tmp["Keywords"].values.tolist(), index=df_isw_tmp.index)

df_isw_v2["report_date"] = df_isw["report_date"]
df_isw_v2["date_tomorrow_datetime"] = df_isw["date_tomorrow_datetime"]
df_isw_v2["date_tomorrow_epoch"] = df_isw["date_tomorrow_epoch"]

df_isw_v2.fillna(0, inplace=True)


del df_isw_tmp
del df_isw

df_isw_v2.head(1)

# %% [markdown]
# ### Adding holidays feature

# %%
add_ukrainian_holidays(
    df_isw_v2, day_datetime_column="report_date", column_name="ukrainian_holiday"
)

df_isw_v2.head(1)

# %% [markdown]
# ### Preparing alarms data

# %%
df_alarms = pd.read_csv("../../external_data/alarms/alarms.csv", sep=";")
df_alarms.head(1)

# %% cell_id="ce00bb16d2e94d128d28f352bde22cbb" deepnote_cell_type="code"
df_alarms.drop(["id", "region_id"], axis=1, inplace=True)
df_alarms["event_time"] = np.nan

# %% cell_id="43771d041c58424eb692f73230f4b791" deepnote_cell_type="code"
df_alarms.head(5)

# %% cell_id="613e468d21b7436dbe18a526bcf8d296" deepnote_cell_type="code"
df_alarms["start_time"] = pd.to_datetime(df_alarms["start"])
df_alarms["end_time"] = pd.to_datetime(df_alarms["end"])
df_alarms["event_time"] = pd.to_datetime(df_alarms["event_time"])

# %% cell_id="3797b89c253e4201969251924a6c753f" deepnote_cell_type="code"
df_alarms["start_hour"] = df_alarms["start_time"].dt.floor("H")
df_alarms["end_hour"] = df_alarms["end_time"].dt.ceil("H")
df_alarms["event_hour"] = df_alarms["event_time"].dt.round("H")

# %% cell_id="fddc3ee690664ca9ab4d0c8dce8a3138" deepnote_cell_type="code"
df_alarms["start_hour"] = df_alarms.apply(
    lambda x: x["start_hour"] if not isNaN(x["start_hour"]) else x["event_hour"], axis=1
)
df_alarms["end_hour"] = df_alarms.apply(
    lambda x: x["end_hour"] if not isNaN(x["end_hour"]) else x["event_hour"], axis=1
)

# %% cell_id="c4638f2955c74e41995e3ddaf348c408" deepnote_cell_type="code"
df_alarms["day_date"] = df_alarms["start_time"].dt.date

df_alarms["start_hour_datetimeEpoch"] = df_alarms["start_hour"].apply(
    lambda x: int(x.timestamp()) if not isNaN(x) else None
)
df_alarms["end_hour_datetimeEpoch"] = df_alarms["end_hour"].apply(
    lambda x: int(x.timestamp()) if not isNaN(x) else None
)

df_alarms.head(5)

# %%
df_alarms_v2 = df_alarms.copy()

# %%
# drop all columns except for (df_events_v2)
df_alarms_v2.drop(
    columns=[
        "start",
        "end",
        "event_time",
        "start_time",
        "end_time",
        "start_hour",
        "end_hour",
        "event_hour",
        "clean_end",
        # keep in mind) we will use this column later
        "all_region",
        "intersection_alarm_id",
    ],
    inplace=True,
)

# %%
df_alarms_v2.head(1)

# %% [markdown] cell_id="60915435755f423ba7455e694efcff97" deepnote_cell_type="markdown"
# ### Prepare weather

# %% cell_id="128e051e53694b4f92b1d765f6e5eb4b" deepnote_cell_type="code"
df_weather = pd.read_csv("../../external_data/hourly_weather/all_weather_by_hour.csv")
df_weather["day_datetime"] = pd.to_datetime(df_weather["day_datetime"])

# %% cell_id="6ab1b03027504c218ed5e02b5944ee28" deepnote_cell_type="code"
df_weather.shape

# %% cell_id="ca2c2b5a13334a78b21504402d417830" deepnote_cell_type="code"
df_weather.head(1)

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
    "hour_preciptype",
]

# %% cell_id="02dffa78339b409eba1906019ae553fe" deepnote_cell_type="code"
df_weather.drop(
    weather_exclude,
    axis=1,
    inplace=True,
)

# %% cell_id="bd136660563e43478b9d934a89d5da1b" deepnote_cell_type="code"
df_weather["city"] = df_weather["city_resolvedAddress"].apply(
    lambda x: x.split(",")[0]
)
df_weather["city"] = df_weather["city"].replace(
    "Хмельницька область", "Хмельницький"
)

# %%
# fill nan values
df_weather.fillna(0, inplace=True)

# %% cell_id="7035ab608f4f440aacd6a30d55f981d8" deepnote_cell_type="code"
df_weather.head(5)

# %% cell_id="2a2e76b0bf174bcb98622170e00d66c4" deepnote_cell_type="code"
df_weather.columns

# %% [markdown] cell_id="12b18e5dfd784c70b4170075378f7080" deepnote_cell_type="markdown"
# ### Merging weather with regions

# %% cell_id="601c4a8529864ae9a16c0630e4e3b64d" deepnote_cell_type="code"
REGIONS_DATA_FOLDER = "../external_data/additions"
REGIONS_DATA_FILE = "regions.csv"
df_regions = pd.read_csv(f"{REGIONS_DATA_FOLDER}/{REGIONS_DATA_FILE}")

# %% cell_id="204c1f5676cb438b87b5523b625886a3" deepnote_cell_type="code"
df_regions.head(5)

# %% cell_id="c07fbbbdcf584e8993cc7d4d4a8b8651" deepnote_cell_type="code"
df_weather_v2 = pd.merge(
    df_weather, df_regions, left_on="city", right_on="center_city_ua"
)

# %% cell_id="44a4b2b9da614135b11dcf0c31ff90cc" deepnote_cell_type="code"
df_weather_v2.head(10)

# %% cell_id="21a4956d1ab040ed947ed172556665b6" deepnote_cell_type="code"
df_weather_v2.shape

# %%
del df_weather

# %% [markdown] cell_id="553fbeced23c40a48a3504d00b1d70e9" deepnote_cell_type="markdown"
# ### Merging weather and alarms

# %% cell_id="40055894945e4707977c52ac7def16b6" deepnote_cell_type="code"
df_alarms_v2.dtypes

# %% cell_id="ca638e87e4ad4d4fb75a1833e5ae7f4c" deepnote_cell_type="code"
df_alarms_v2.shape

# %% cell_id="6d40bd1596004f1abef8ccf066f59e9d" deepnote_cell_type="code"
df_alarms_v2.head(10)

# %% cell_id="f563ac4a436740d1a1e69e78e4a4b3a2" deepnote_cell_type="code"
events_dict = df_alarms.to_dict("records")
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
df_alarms_v3 = pd.DataFrame.from_dict(events_by_hour)
df_alarms_v3["hour_level_event_datetimeEpoch"] = df_alarms_v3[
    "hour_level_event_time"
].apply(lambda x: int(x.timestamp()) if not isNaN(x) else None)

# %%
del df_alarms

# %% cell_id="f9e6d8dcd0c24da882366d0919c71b95" deepnote_cell_type="code"
df_alarms_v3.shape

# %% cell_id="8e4a4854d0c1438c8f6ea96d590648cf" deepnote_cell_type="code"
df_alarms_v3.head(1)

# %% cell_id="ff28d9be8f154e029b442fee3319b359" deepnote_cell_type="code"
df_weather_v2.shape

# %% cell_id="e6dbcef495b24bcfa953dc2bba6b35cb" deepnote_cell_type="code"
df_weather_v2.head(1)

# %% cell_id="ec545fd52e5c40a7a58a7293301fa52b" deepnote_cell_type="code"
df_alarms_v4 = df_alarms_v3.copy().add_prefix("event_")
df_alarms_v4.head(1)

# %%
del df_alarms_v3

# %% cell_id="c196ef5769f14944b471c1e9c8182dbd" deepnote_cell_type="code"
df_weather_v3 = df_weather_v2.merge(
    df_alarms_v4,
    how="left",
    left_on=["region_alt", "hour_datetimeEpoch"],
    right_on=["event_region_title", "event_hour_level_event_datetimeEpoch"],
)

# %%
del df_weather_v2

# %% cell_id="db4146abf636459a9947e6ae334ec013" deepnote_cell_type="code"
# Alarm data
print(df_weather_v3.loc[~isNaN(df_weather_v3["event_start"])].shape)
print(df_weather_v3.loc[isNaN(df_weather_v3["event_start"])].shape)
df_weather_v3["is_alarm"] = df_weather_v3.apply(
    lambda x: 0 if isNaN(x["event_start"]) else 1, axis=1
)
no_alarms = df_weather_v3.loc[df_weather_v3["is_alarm"] == 0].size
alarms = df_weather_v3.loc[df_weather_v3["is_alarm"] == 1].size
print(f"Alarm chane: {alarms / df_weather_v3.size}")
print(f"No alarm: {no_alarms / df_weather_v3.size}")

# %% [markdown] cell_id="ad9519e8c60a4645ad9d5159748f1dd6" deepnote_cell_type="markdown"
# ### Number of alarms for this region during the last 24 hours

# %%
df_weather_v4 = None

# %% cell_id="3c0032ffd14144caab2b1bed8db9ab1b" deepnote_cell_type="code" language="sql"
# df_weather_v4 << select df.*, coalesce(alarm_count.events_last_24_hrs, 0) as events_last_24_hrs
# from df_weather_v3 df
#          left join (select out.region_id,
#                             out.hour_datetimeEpoch,
#                             count(inn.event_start_time) as events_last_24_hrs
#                      from df_weather_v3 out
#                               left join df_weather_v3 inn
#                                          on out.region_id = inn.region_id
#                      where
#                        inn.event_start_time::timestamp
#                         between
#                         (epoch_ms(out.hour_datetimeEpoch::long * 1000) - '24 HOURS'::interval)
#                             and epoch_ms(out.hour_datetimeEpoch::long * 1000)
#                      group by out.region_id, out.hour_datetimeEpoch) as alarm_count
#         on df.region_id = alarm_count.region_id and df.hour_datetimeEpoch = alarm_count.hour_datetimeEpoch;

# %% cell_id="f2996c9e0cf84c1595b02ae9db365503" deepnote_cell_type="code"
df_weather_v4[
    ["city_resolvedAddress", "region_id", "hour_datetimeEpoch", "events_last_24_hrs"]
].tail(5)

df_weather_v4.drop(["city_resolvedAddress"], axis=1, inplace=True)

# %% cell_id="7a80f3a8324f4f759fe717196aa8b9e3" deepnote_cell_type="code"
# Add day of week name
df_weather_v4["day_of_week"] = df_weather_v4["day_datetime"].apply(
    lambda date: pd.to_datetime(date).day_name()
)

# %%
df_weather_v4[["day_datetime", "day_of_week"]].tail(15)

# %%
df_weather_v4.head(1)

# %% [markdown]
# ### Adding lunar and solar features

# %%
df_weather_v5 = None

# %% language="sql"
# df_weather_v5 << select df.*, coalesce(alarmed_count.alarmed_regions_count, 0) as alarmed_regions_count
# from df_weather_v4 df
#          left join (select out.day_datetimeEpoch,
#                                             out.hour_datetimeEpoch,
#                                             count(*) as alarmed_regions_count
#                                     from df_weather_v4 out left join df_weather_v4 inn
#                                          on out.day_datetimeEpoch = inn.day_datetimeEpoch and out.hour_datetimeEpoch = inn.hour_datetimeEpoch
#                                     where inn.is_alarm = 1 and inn.hour_datetimeEpoch
#                                                                between out.event_start_hour_datetimeEpoch and out.event_start_hour_datetimeEpoch
#                                     group by out.day_datetimeEpoch, out.hour_datetimeEpoch) as alarmed_count
#         on df.day_datetimeEpoch = alarmed_count.day_datetimeEpoch and df.hour_datetimeEpoch = alarmed_count.hour_datetimeEpoch;

# %%
del df_weather_v4

# %%
add_lunar_eclipses(df_weather_v5, day_datetime_column="day_datetime")

# %%
df_weather_v5.head(1)

# %% [markdown]
# ### Final merge preparations

# %%
df_weather_v5.shape

# %%
# from typing import Tuple
#
# # Defining the function for dummy creation
# def create_dummy(df: pd.DataFrame, dummy_var_list: list) -> Tuple:
#     """
#     Creates dummy variables for the variables in dummy_var_list
#     Returns a tuple of the following
#         * df - The dataframe with the dummy variables
#         * dummy_var_list - The list of dummy variables created
#     """
#     # Placeholder for the dummy variables
#     added_features = []
#     for var in dummy_var_list:
#         dummy = pd.get_dummies(df[var], prefix=var, drop_first=False)
#
#         # Adding the new features to list
#         added_features.extend(dummy.columns)
#
#         # Adding the dummy variables to the dataframe
#         df = pd.concat([df, dummy], axis=1)
#         df.drop(var, axis=1, inplace=True)
#
#     # Returning the dataframe
#     return df, added_features

# %%
# df_weather_v6, categorical_features = create_dummy(df_weather_v5.copy(), ["day_of_week", "hour_conditions", "region_id"])
# print(categorical_features)
# print(len(categorical_features))

# %%
print(df_weather_v5.shape)
df_weather_v5.drop(['day_precipcover'], axis=1, inplace=True)
df_weather_v5.head(5)

# %%
df_weather_v5 = df_weather_v5.sort_index(axis=1)
df_weather_v5.columns


# %%
df_weather_v5.head(1)

# %%
# fill NaN values with 0
# df_weather_v6.fillna(0, inplace=True)

# %%
# drop all columns that are of type

# %%
# drop columns
# TODO: revert back dropped columns
df_weather_v5.drop(
    [
        "event_start_time",
        "event_end_time",
        "event_start_hour",
        "event_end_hour",
        "event_hour_level_event_time",

        "event_intersection_alarm_id",
        "event_start_hour_datetimeEpoch",
        "event_end_hour_datetimeEpoch",
        "event_hour_level_event_datetimeEpoch",

        "event_all_region"
    ],
    axis=1,
    inplace=True,
)

# %%
# view its types
df_weather_v5.dtypes

# %%
df_weather_v5["day_datetimeEpoch"].head(15)

# %%
# drop from isw_v2
# df_isw_v2.drop(["report_date", "date_tomorrow_datetime"], axis=1, inplace=True)
df_isw_v2.drop(["report_date"], axis=1, inplace=True)

# %%
df_weather_v5.shape

# %% [markdown]
# ### Final merge

# %%
# df_isw_v2["date_tomorrow_epoch"].head(15)
print(df_weather_v5.shape)
print(df_isw_v2.shape)

# %%
df_weather_v5.head(1)

# %%
# df_weather_v6.drop(['hour_sunset','hour_sunrise', 'hour_preciptype'], axis=1, inplace=True)
df_weather_v5 = df_weather_v5.sort_index(axis=1)

df_weather_v5.columns


# %%
# separate the target variable from the features
target = df_weather_v5.pop('is_alarm')
day_datetimeEpoch = df_weather_v5.pop('day_datetimeEpoch')
hour_datetimeEpoch = df_weather_v5.pop('hour_datetimeEpoch')
alarmed_regions_count = df_weather_v5.pop('alarmed_regions_count')
events_last_24_hrs = df_weather_v5.pop('events_last_24_hrs')
day_datetime = df_weather_v5.pop('day_datetime')
day_of_week = df_weather_v5.pop("day_of_week")
hour_conditions = df_weather_v5.pop("hour_conditions")
region_id = df_weather_v5.pop("region_id")

# %%
df_weather_v5 = df_weather_v5.select_dtypes(exclude=['object'])

# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler



# create a StandardScaler object and fit it to the features
scaler = StandardScaler()
scaler.fit(df_weather_v5)

# transform the features using the scaler
df_scaled = scaler.transform(df_weather_v5)
from sklearn.preprocessing import MinMaxScaler

scaler2 = MinMaxScaler()
minmax_df = np.array(df_scaled)
minmax_df = scaler2.fit_transform(minmax_df)

# combine the scaled features and the target variable into a new dataframe
df_weather_v5 = pd.DataFrame(minmax_df, columns=df_weather_v5.columns)
df_weather_v5['is_alarm'] = target
df_weather_v5['day_datetimeEpoch'] = day_datetimeEpoch
df_weather_v5['hour_datetimeEpoch'] = hour_datetimeEpoch
df_weather_v5['day_datetime'] = day_datetime
df_weather_v5['day_of_week'] = day_of_week
df_weather_v5['hour_conditions'] = hour_conditions
df_weather_v5['region_id'] = region_id
df_weather_v5['alarmed_regions_count'] = alarmed_regions_count
df_weather_v5['events_last_24_hrs'] = events_last_24_hrs

# %%
df_weather_v5.head(1)

# %%
from typing import Tuple


# Defining the function for dummy creation
def create_dummy(df: pd.DataFrame, dummy_var_list: list) -> Tuple:
    """
    Creates dummy variables for the variables in dummy_var_list
    Returns a tuple of the following
        * df - The dataframe with the dummy variables
        * dummy_var_list - The list of dummy variables created
    """
    # Placeholder for the dummy variables
    added_features = []
    for var in dummy_var_list:
        dummy = pd.get_dummies(df[var], prefix=var, drop_first=False)

        # Adding the new features to list
        added_features.extend(dummy.columns)

        # Adding the dummy variables to the dataframe
        df = pd.concat([df, dummy], axis=1)
        df.drop(var, axis=1, inplace=True)

    # Returning the dataframe
    return df, added_features



# %%
print(df_weather_v5.dtypes)


# %%
df_weather_v6, categorical_features = create_dummy(df_weather_v5.copy(),
                                                   ["day_of_week", "hour_conditions"])
print(categorical_features)
print(len(categorical_features))

# %%
df_weather_v6.columns

# %%
# alarmed_regions_count           float64
# day_dew                         float64
# day_humidity                    float64
# day_moonphase                   float64
# day_precip                      float64
# day_solarenergy                 float64
# day_solarradiation              float64
# day_temp                        float64
# day_tempmax                     float64
# day_tempmin                     float64
# day_uvindex                     float64
# events_last_24_hrs              float64
# hour_cloudcover                 float64
# hour_dew                        float64
# hour_humidity                   float64
# hour_precip                     float64
# hour_precipprob                 float64
# hour_pressure                   float64
# hour_severerisk                 float64
# hour_snow                       float64
# hour_snowdepth                  float64
# hour_solarenergy                float64
# hour_solarradiation             float64
# hour_temp                       float64
# hour_uvindex                    float64
# hour_visibility                 float64
# hour_winddir                    float64
# hour_windgust                   float64
# hour_windspeed                  float64
# lunar_eclipse                   float64
# moonphased_eclipse              float64
# is_alarm                          int64
# day_datetimeEpoch                 int64
# hour_datetimeEpoch                int64
# day_datetime             datetime64[ns]
# day_of_week                      object
# hour_conditions                  object
# region_id                         int64
#
#
# alarmed_regions_count    float64
# day_dew                  float64
# day_humidity             float64
# day_moonphase            float64
# day_precip               float64
# day_solarenergy          float64
# day_solarradiation       float64
# day_temp                 float64
# day_tempmax              float64
# day_tempmin              float64
# day_uvindex              float64
# events_last_24_hrs       float64
# hour_cloudcover          float64
# hour_dew                 float64
# hour_humidity            float64
# hour_precip              float64
# hour_precipprob          float64
# hour_pressure            float64
# hour_severerisk          float64
# hour_snow                float64
# hour_snowdepth           float64
# hour_solarenergy         float64
# hour_solarradiation      float64
# hour_temp                float64
# hour_uvindex             float64
# hour_visibility          float64
# hour_winddir             float64
# hour_windgust            float64
# hour_windspeed           float64
# lunar_eclipse            float64
# moonphased_eclipse       float64
# day_datetimeEpoch          int64
# hour_datetimeEpoch         int64
# day_of_week               object
# hour_conditions           object
# region_id                  int64

# %%

# %%
df_fin = pd.merge(
    df_weather_v6,
    df_isw_v2,
    left_on="day_datetime",
    right_on="date_tomorrow_datetime",
)
# missing data
# df_fin_2 = pd.merge(
#     df_weather_v6,
#     df_isw_v2,
#     left_on="day_datetimeEpoch",
#     right_on="date_tomorrow_epoch",
# )
df_fin.drop(["day_datetime", "date_tomorrow_datetime"], axis=1, inplace=True)
# df_fin_2.drop(["day_datetime", "date_tomorrow_datetime"], axis=1, inplace=True)

# %%
df_weather_v6.drop(["day_datetime"], axis=1, inplace=True)

df_weather_v6.to_pickle(f"{OUTPUT_FOLDER}/df_fin_history_no_isw.pkl")


# %%
print(df_fin.shape)
# print(df_fin_2.shape)

# %%
# df_fin nan to 0
df_fin.fillna(0, inplace=True)

# %%
df_fin.head(15)

# %% [markdown] cell_id="69c38d2760aa4198b6a49ea956f06e52" deepnote_cell_type="markdown"
# ### Save final merged dataframe

# %% cell_id="8b906aa46c70419b8c5a2aeb4be211e0" deepnote_cell_type="code"
df_fin.to_pickle(f"{OUTPUT_FOLDER}/df_fin_history.pkl")


# %%
print(df_fin.dtypes.count)


# %%
df_fin.head(10)


# %%
