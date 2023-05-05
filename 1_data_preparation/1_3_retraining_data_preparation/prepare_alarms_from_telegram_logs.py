# %%
import pandas as pd
import numpy as np

# %%
# Load JSON data from a file
with open('messages.json', 'r', encoding='utf-8') as f:
    data = f.read()

# Convert JSON data to a DataFrame
source_log = pd.read_json(data)
source_log = source_log[source_log['date'] >= '2023-01-20']

# %%
source_log.head(10000)

# %%
source_log = source_log[~source_log['message'].str.contains('🟡')]
source_log = source_log[~source_log['message'].str.contains(' м. ')]

source_log = source_log[source_log['message'].str.contains('Відбій|Повітряна')]

source_log['region_name'] = source_log['message'].apply(lambda x: x.split('#')[1])
source_log.head(40)

# %%
source_log.loc[source_log['message'].str.contains('Відбій'), 'event'] = 'Відбій'
source_log.loc[source_log['message'].str.contains('Повітряна'), 'event'] = 'Повітряна'


# %% [markdown]
# source_log['region'] = source_log['message'].apply(lambda x: x.split('#')[1])

# %%
source_log = source_log[source_log['region_name'].str.contains('область')]

# %%
source_log['region_name'] = source_log['region_name'].apply(lambda x: x.split('_')[0])

# %%
source_log.loc[source_log['region_name'] == 'ІваноФранківська', 'region_name'] = 'Івано-Франківська'
source_log_sorted = source_log.groupby('region_name').apply(lambda x: x.sort_values('date'))
source_log_sorted.drop(['region_name'], axis=1, inplace=True)
source_log_sorted.head(1100)


# %%
# Add a new column with the 'Age' value of the next row
source_log_sorted['end'] = source_log_sorted['date'].shift(-1).where(source_log_sorted['event'].str.contains('Повітряна'), other=np.nan)

# Drop the last row of the DataFrame, which has a NaN value for the 'Next_Age' column
source_log_sorted.drop(source_log_sorted.tail(1).index, inplace=True)


# %%
source_log_sorted = source_log_sorted.dropna(subset=['end'], how='any')
source_log_sorted.drop(['event', 'message'], axis=1, inplace=True)
source_log_sorted = source_log_sorted.rename(columns={'date': 'start'})
source_log_sorted.head(1100)


# %%
source_log_sorted = source_log_sorted.reset_index()

# %%
df_regions = pd.read_csv("../../external_data/additions/regions.csv", sep=",")

df_regions.drop(['region_alt'], axis=1, inplace=True)

df_regions.drop(df_regions[df_regions['region'] == 'АР Крим'].index, inplace=True)
df_regions.drop(df_regions[df_regions['region'] == 'Луганська'].index, inplace=True)

df_regions.head(5)

# %%
df_fin_alarms = source_log_sorted.merge(
    df_regions,
    how="right",
    left_on=["region_name"],
    right_on=["region"],
)
df_fin_alarms['region_title'] = df_fin_alarms['region_name']
df_fin_alarms = df_fin_alarms.rename(columns={'center_city_ua': 'region_city'})
df_fin_alarms.drop(['level_1','region','region_name','center_city_en'],axis=1, inplace=True)

df_fin_alarms['all_region'] = 0
df_fin_alarms['clean_end'] = df_fin_alarms['end']
df_fin_alarms['intersection_alarm_id'] = np.nan

df_fin_alarms.head(1100)


# %%

# %%
df_fin_alarms.to_csv("./results/df_fin_alarms.csv")


# %%
