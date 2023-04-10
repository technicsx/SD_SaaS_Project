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
import os

import pandas as pd
import seaborn as sns

from paths_full import DATA_PREP_RESULTS_FOLDER

pickle_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7.pkl")
df_final = pd.read_pickle(pickle_path)

# %%
corr = df_final.corr()[['is_alarm']].sort_values(by='is_alarm', ascending=False).head(20)
print(corr)
sns.heatmap(corr, annot=True, cmap='coolwarm')

# %% [markdown]
# ### Feature Analysis

# %%
df_final[['events_last_24_hrs', 'alarmed_regions_count']].describe()

# %%
filter_col = [col for col in df_final if col.startswith('day_of_week_')]
df_final_days_of_week = df_final[["is_alarm"] + filter_col]
corr = df_final_days_of_week.corr()[['is_alarm']].sort_values(by='is_alarm', ascending=False)
sns.heatmap(corr, annot=True, cmap='coolwarm')
corr

# %%
filter_col = [col for col in df_final if col.startswith('hour_conditions_')]
df_final_days_of_week = df_final[["is_alarm"] + filter_col]
corr = df_final_days_of_week.corr()[['is_alarm']].sort_values(by='is_alarm', ascending=False)
sns.heatmap(corr, annot=True, cmap='coolwarm')
corr

# %%
filter_col = [col for col in df_final if col.startswith('city_')]
df_final_days_of_week = df_final[["is_alarm"] + filter_col]
city_corr = df_final_days_of_week.corr()[['is_alarm']].sort_values(by='is_alarm', ascending=False)
# sns.heatmap(corr, annot=True)
city_corr

# %%
df_final_holidays = df_final[
    [
        "is_alarm",
        'ukrainian_holiday',
        'russian_holiday',
        'lunar_eclipse',
        'solar_eclipse'
    ]
]
corr = df_final_holidays.corr()[['is_alarm']].sort_values(by='is_alarm', ascending=False)
# sns.heatmap(corr, annot=True)
corr

# %%
corr = df_final.corr()[['ukrainian_holiday']].sort_values(by='ukrainian_holiday', ascending=False).head(10)
# sns.heatmap(corr, annot=True)
corr

# %%
