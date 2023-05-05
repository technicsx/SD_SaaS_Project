# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# !pip install pandas
# !pip install numpy

# %%
import datetime
import numpy as np
import pandas as pd
import pickle
import os

# %%
df_isw = pd.read_csv("./results/tfidf-new.csv", sep=",")
df_isw.head(5)


# %%
from src.features.holidays_feature import add_ukrainian_holidays
from src.features.eclipses_feature import add_lunar_eclipses

# %%
df_isw["report_date"] = pd.to_datetime(df_isw["Date"])
df_isw.drop(columns=["Date", "Name"], inplace=True)
df_isw.head(1)

df_isw["date_tomorrow_datetime"] = df_isw["report_date"].apply(
    lambda x: x + datetime.timedelta(days=1)
)
df_isw["date_tomorrow_datetime"] = df_isw["date_tomorrow_datetime"]
df_isw["date_tomorrow_epoch"] = pd.to_datetime(df_isw["date_tomorrow_datetime"])
df_isw['date_tomorrow_epoch'] = df_isw['date_tomorrow_datetime'].apply(lambda x: x.timestamp())
# df_isw["date_tomorrow_epoch"] = df_isw["date_tomorrow_epoch"] - 10800

df_isw["event_time"] = np.nan
df_isw.head(1)

df_isw_tmp = df_isw.copy()
df_isw_tmp["Keywords"] = df_isw_tmp["Keywords"].apply(lambda x: dict(eval(x)))

df_isw_v2 = pd.DataFrame(df_isw_tmp["Keywords"].values.tolist(), index=df_isw_tmp.index)

df_isw_v2["report_date"] = df_isw["report_date"]
df_isw_v2["date_tomorrow_datetime"] = pd.to_datetime(df_isw["date_tomorrow_datetime"])
df_isw_v2["date_tomorrow_epoch"] = df_isw["date_tomorrow_epoch"]

df_isw_v2.fillna(0, inplace=True)

del df_isw_tmp
del df_isw

df_isw_v2.head(1)

# %%
add_ukrainian_holidays(
    df_isw_v2, day_datetime_column="report_date", column_name="ukrainian_holiday"
)

# %%
df_alarms = pd.read_csv("../data/alarms/alarms_15_00.csv", sep=",")
df_alarms.head(5)


# %%
df_weather = pd.read_csv("../data/weather/weather_18_00.csv", sep=",")
df_weather.head(5)


# %%
df_weather.rename(columns={'datetime': 'day_datetimeEpoch'}, inplace=True)
df_weather.rename(columns={'wgust': 'windgust'}, inplace=True)
df_weather.rename(columns={'wspd': 'windspeed'}, inplace=True)
df_weather.rename(columns={'wdir': 'winddir'}, inplace=True)
df_weather.drop(['windchill', 'cin'], axis=1, inplace=True)
df_weather.rename(columns={'pop': 'hour_precipprob'}, inplace=True)
df_weather.rename(columns={'sealevelpressure': 'pressure'}, inplace=True)
df_weather.head(5)

# %%
df_weather['day_datetime'] = pd.to_datetime(df_weather['datetimeStr'])


# %%
df_weather.head(5)


# %%
df_weather['day_datetime_epoch'] = df_weather['day_datetime'].dt.date

# %%
df_weather.head(5)


# %%
df_weather["day_datetime_epoch"] = df_weather["day_datetime_epoch"]

# df_weather['day_datetime_epoch'] = pd.to_datetime(df_weather['day_datetime_epoch']).apply(lambda x: x.replace(year=2022))
df_weather.rename(columns={'day_datetimeEpoch': 'hour_datetimeEpoch'}, inplace=True)
df_weather['day_datetimeEpoch'] = pd.to_datetime(df_weather['day_datetime_epoch']).apply(lambda x: x.timestamp())

# extract only the date part from the 'datetime' column
# df_weather['day_datetime_epoch'] = df_weather['day_datetime_epoch'].apply(lambda x: x.timestamp())

# %%
df_weather.head(5)

# %%
df_weather['day_tempmax'] = df_weather.groupby(['city_name','day_datetime'])['temp'].transform('max')
df_weather['day_tempmin'] = df_weather.groupby(['city_name','day_datetime'])['temp'].transform('min')
df_weather['day_temp'] = df_weather.groupby(['city_name','day_datetime'])['temp'].transform('mean').round(1)

df_weather['day_snow'] = df_weather.groupby(['city_name','day_datetime'])['snow'].transform('mean').round(1)
df_weather['day_visibility'] = df_weather.groupby(['city_name','day_datetime'])['visibility'].transform('mean').round(1)
df_weather['day_solarradiation'] = df_weather.groupby(['city_name','day_datetime'])['solarradiation'].transform('mean').round(1)
df_weather['day_solarenergy'] = df_weather.groupby(['city_name','day_datetime'])['solarenergy'].transform('mean').round(1)
df_weather['day_uvindex'] = df_weather.groupby(['city_name','day_datetime'])['uvindex'].transform('mean').round(1)
df_weather['day_severerisk'] = df_weather.groupby(['city_name','day_datetime'])['severerisk'].transform('mean').round(1)
df_weather['day_moonphase'] = df_weather.groupby(['city_name','day_datetime'])['moonphase'].transform('mean')
df_weather['day_humidity'] = df_weather.groupby(['city_name','day_datetime'])['humidity'].transform('mean').round(2)
df_weather['day_dew'] = df_weather.groupby(['city_name','day_datetime'])['dew'].transform('mean').round(1)
df_weather['day_precip'] = df_weather.groupby(['city_name','day_datetime'])['precip'].transform('mean').round(1)

df_weather.head(5)


# %%
df_weather.columns = [
    'hour_' + col if not col.startswith('day_') and not col.startswith('city_') and not col.startswith('hour_') else col
    for col in df_weather.columns]

# %%
df_weather.head(5)


# %%
add_lunar_eclipses(df_weather, day_datetime_column="day_datetime")

# %%
df_weather_K = pd.read_csv("../external_data/hourly_weather/all_weather_by_hour.csv", sep=",")
df_weather_K.head(5)

# %%
df_weather_K.drop(['city_latitude', 'city_longitude', 'city_timezone', 'city_tzoffset', 'day_datetime'], axis=1,
                  inplace=True)


# %%
df_regions = pd.read_csv("../external_data/additions/regions.csv", sep=",")

df_regions.drop(['center_city_ua', 'region_alt'], axis=1, inplace=True)

# %%
df_regions.head(5)


# %%
df_regions.drop(df_regions[df_regions['region'] == 'АР Крим'].index, inplace=True)
df_regions.drop(df_regions[df_regions['region'] == 'Луганська'].index, inplace=True)


# %%
df_regions.head(5)


# %%
df_merged_weather_with_regions = df_weather.merge(
    df_regions,
    how="right",
    left_on=["city_name"],
    right_on=["center_city_en"],
)

df_merged_weather_with_regions.head(1)

# %%
# df_alarms["start_hour_epoch"] = pd.to_datetime(df_alarms["start_hour"])
# df_alarms['start_hour_epoch'] = df_alarms['start_hour_epoch'].apply(lambda x: int(x.timestamp() * 1000))
df_alarms.drop(['start_hour', 'ua_alarms_region_id'], axis=1, inplace=True)
df_merged_weather_with_regions.drop(
    ['day_datetime', 'center_city_en', 'day_datetime_epoch', 'hour_sunrise', 'hour_sunset', 'hour_preciptype',
     'hour_heatindex'], axis=1, inplace=True)
df_alarms.head(100)

# %%
df_merged_weather_with_regions_with_alarms = df_merged_weather_with_regions.merge(
    df_alarms,
    how="left",
    left_on=["region"],
    right_on=["region"]
)

# Add the 'isAlarm' column to the dataframe
# df_merged_weather_with_regions_with_alarms['is_alarm'] = is_alarm
df_merged_weather_with_regions_with_alarms.head(5)

# %%

# %%
df_a = df_merged_weather_with_regions_with_alarms[
    df_merged_weather_with_regions_with_alarms['day_datetimeEpoch'] == '1682262000000']

# Output the selected rows
df_a

# %%
df_isw_v2.head(1)

# %%

# %%
df_merged_weather_with_regions_with_alarms.drop(['city_name', 'region'], axis=1, inplace=True)
df_merged_weather_with_regions_with_alarms.head(1)

# %%
df_merged_weather_with_regions_with_alarms["day_of_week"] = df_merged_weather_with_regions_with_alarms[
    "day_datetime"].apply(
    lambda date: pd.to_datetime(date).day_name()
)

# %%
df_merged_weather_with_regions_with_alarms.head(1)

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
def oneHotDayOfWeek(df):
    one_hot = pd.get_dummies(df['day_of_week'], prefix='day_of_week', drop_first=False)

    # concatenate the one-hot encoding to the original dataframe
    df_encoded = pd.concat([df, one_hot], axis=1)

    # set all the other day_of_week columns as False
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days_of_week:
        if day not in df_encoded.columns:
            df_encoded['day_of_week_' + day] = False

    # re-order the columns
    return df_encoded


# %%
def oneHotHourCondition(df):
    one_hot = pd.get_dummies(df['hour_conditions'], prefix='hour_conditions', drop_first=False)

    # concatenate the one-hot encoding to the original dataframe
    df_encoded = pd.concat([df, one_hot], axis=1)

    # set all the other day_of_week columns as False
    hour_conds = ['Clear', 'Freezing Drizzle/Freezing Rain, Overcast', 'Ice, Overcast', 'Overcast', 'Partially cloudy',
                  'Rain', 'Rain, Overcast', 'Rain, Partially cloudy', 'Snow', 'Snow, Overcast',
                  'Snow, Partially cloudy', 'Snow, Rain', 'Snow, Rain, Overcast', 'Snow, Rain, Partially cloudy']
    for cond in hour_conds:
        if cond not in df_encoded.columns:
            df_encoded['hour_conditions_' + cond] = False

    # re-order the columns
    return df_encoded


# %%
def oneHotRegionId(df):
    one_hot = pd.get_dummies(df['region_id'], prefix='region_id', drop_first=False)

    # concatenate the one-hot encoding to the original dataframe
    df_encoded = pd.concat([df, one_hot], axis=1)

    # set all the other day_of_week columns as False
    region_ids = [10,
                  11, 13, 14, 15,
                  16, 17, 18, 19,
                  2, 20, 21, 22,
                  23, 24, 25,
                  4, 5, 6, 7,
                  8, 9]
    for id in region_ids:
        if id not in df_encoded.columns:
            df_encoded['region_id_' + str(id)] = False

    # re-order the columns
    return df_encoded

# %%
# df_merged_weather_with_regions_with_alarms_v2, categorical_features = create_dummy(df_merged_weather_with_regions_with_alarms.copy(), ["day_of_week", "hour_conditions"])



# %%
df_merged_weather_with_regions_with_alarms.head(300)

# %%
df_merged_weather_with_regions_with_alarms.drop(
    ['day_severerisk', 'day_snow', 'hour_cape', 'hour_datetimeStr', 'hour_moonphase',
     'day_visibility'], axis=1, inplace=True)
df_merged_weather_with_regions_with_alarms_v2 = df_merged_weather_with_regions_with_alarms.sort_index(axis=1)
df_merged_weather_with_regions_with_alarms_v2.columns

# %%
df_merged_weather_with_regions_with_alarms_v2['day_datetime'] = pd.to_datetime(
    df_merged_weather_with_regions_with_alarms_v2['day_datetime'])
df_merged_weather_with_regions_with_alarms_v2['day_datetimeEpoch'] = df_merged_weather_with_regions_with_alarms_v2[
    'day_datetimeEpoch'].astype('int64')
df_merged_weather_with_regions_with_alarms_v2.dtypes

# %%
df_merged_weather_with_regions_with_alarms_v2.drop(["day_datetime"], axis=1, inplace=True)

import pandas as pd
from sklearn.preprocessing import StandardScaler

# separate the target variable from the features
day_datetimeEpoch = df_merged_weather_with_regions_with_alarms_v2.pop('day_datetimeEpoch')
hour_datetimeEpoch = df_merged_weather_with_regions_with_alarms_v2.pop('hour_datetimeEpoch')
day_of_week = df_merged_weather_with_regions_with_alarms_v2.pop("day_of_week")
hour_conditions = df_merged_weather_with_regions_with_alarms_v2.pop("hour_conditions")
region_id = df_merged_weather_with_regions_with_alarms_v2.pop("region_id")
events_last_24_hrs = df_merged_weather_with_regions_with_alarms_v2.pop("events_last_24_hrs")
alarmed_regions_count = df_merged_weather_with_regions_with_alarms_v2.pop("alarmed_regions_count")


# create a StandardScaler object and fit it to the features
scaler = StandardScaler()
scaler.fit(df_merged_weather_with_regions_with_alarms_v2)

# transform the features using the scaler
df_scaled = scaler.transform(df_merged_weather_with_regions_with_alarms_v2)
from sklearn.preprocessing import MinMaxScaler

scaler2 = MinMaxScaler()
minmax_df = np.array(df_scaled)
minmax_df = scaler2.fit_transform(minmax_df)
# combine the scaled features and the target variable into a new dataframe
df_merged_weather_with_regions_with_alarms_v2 = pd.DataFrame(minmax_df,
                                                             columns=df_merged_weather_with_regions_with_alarms_v2.columns)
df_merged_weather_with_regions_with_alarms_v2['day_datetimeEpoch'] = day_datetimeEpoch
df_merged_weather_with_regions_with_alarms_v2['hour_datetimeEpoch'] = hour_datetimeEpoch
df_merged_weather_with_regions_with_alarms_v2['day_of_week'] = day_of_week
df_merged_weather_with_regions_with_alarms_v2['hour_conditions'] = hour_conditions
df_merged_weather_with_regions_with_alarms_v2['region_id'] = region_id
df_merged_weather_with_regions_with_alarms_v2['alarmed_regions_count'] = alarmed_regions_count
df_merged_weather_with_regions_with_alarms_v2['events_last_24_hrs'] = events_last_24_hrs

# %%
print(df_merged_weather_with_regions_with_alarms_v2.dtypes.count)


# %%
df_merged_weather_with_regions_with_alarms_v2 = oneHotDayOfWeek(df_merged_weather_with_regions_with_alarms_v2)
df_merged_weather_with_regions_with_alarms_v2 = oneHotHourCondition(df_merged_weather_with_regions_with_alarms_v2)
# df_merged_weather_with_regions_with_alarms_v2 = oneHotRegionId(df_merged_weather_with_regions_with_alarms_v2)

# %%
df_merged_weather_with_regions_with_alarms_v2.head(23)

# %%
df_merged_weather_with_regions_with_alarms_v2.drop(
    ['day_of_week', 'hour_conditions'], axis=1, inplace=True)

# %%
print(df_merged_weather_with_regions_with_alarms_v2.dtypes)

# %%
# df_merged_weather_with_regions_with_alarms_with_isw = df_merged_weather_with_regions_with_alarms.merge(
#     df_isw_v2,
#     how="left",
#     left_on=["day_datetime_epoch"],
#     right_on=["date_tomorrow_epoch"],
# )
# df_merged_weather_with_regions_with_alarms_with_isw = pd.merge(df_merged_weather_with_regions_with_alarms, df_isw_v2, on='A', how='inner')
# df_merged_weather_with_regions_with_alarms_with_isw['date_tomorrow_epoch'] = df_merged_weather_with_regions_with_alarms_with_isw['date_tomorrow_epoch'].astype('int64')
df_merged_weather_with_regions_with_alarms_with_isw = pd.merge(df_merged_weather_with_regions_with_alarms_v2, df_isw_v2,
                                                               how='cross')

# %%
df_merged_weather_with_regions_with_alarms_with_isw.head(1)


# %%

# %%
nan_rows = df_merged_weather_with_regions_with_alarms_with_isw[
    df_merged_weather_with_regions_with_alarms_with_isw['date_tomorrow_epoch'].isna()]

# %%
nan_rows.head(10)

# %%
df_merged_weather_with_regions_with_alarms_with_isw = df_merged_weather_with_regions_with_alarms_with_isw.sort_index(
    axis=1)

# %%

# %%
df_merged_should_be = pd.read_pickle("./results/df_fin_history.pkl")
df_merged_should_be.head(5)


# %%

df_merged_weather_with_regions_with_alarms_v2.to_pickle("results/df_fin_new_no_isw.pkl")


# %%
df_merged_weather_with_regions_with_alarms_v2.head(10)

# %%
df_merged_weather_with_regions_with_alarms_with_isw.dtypes

# %%
df_merged_weather_with_regions_with_alarms_with_isw['hour_datetimeEpoch']

# %%
df_merged_weather_with_regions_with_alarms_with_isw.columns


# %%
df_merged_weather_with_regions_with_alarms_with_isw.head(100)

# %%
df_merged_weather_with_regions_with_alarms_with_isw.drop(["date_tomorrow_datetime", "report_date"], axis=1,
                                                         inplace=True)

df_merged_weather_with_regions_with_alarms_with_isw.to_pickle("results/df_fin_new.pkl")


# %%
print(df_merged_weather_with_regions_with_alarms_with_isw.dtypes)


# %%

# %%

# %%
