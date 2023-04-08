import numpy as np
import pandas as pd
import os

from paths_full import DATA_PREP_RESULTS_FOLDER

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


csv_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7.csv")
df_final = pd.read_csv(csv_path)


# Fill NaN values
df_final[["event_all_region"]] = df_final[["event_all_region"]].fillna(value=0)

X = np.array(
    df_final[
        [
            "region_id",
            "event_all_region",
            "day_datetimeEpoch",
            "hour_datetimeEpoch",
            "ukrainian_holiday",
            "russian_holiday",
            "hour_temp",
            "hour_snow",
            "hour_visibility",
            "hour_conditions_code",
            "lunar_eclipse",
            "solar_eclipse",
            "moonphased_eclipse",
            "alarmed_regions_count",
        ]
    ]
)
y = np.array(df_final["is_alarm"])

# Dropping any rows with Nan values
# df_final.dropna(inplace = True)

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101
)

regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
