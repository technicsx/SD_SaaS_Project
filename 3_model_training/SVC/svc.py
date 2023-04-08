import numpy as np
import pandas as pd
import os

from paths_full import DATA_PREP_RESULTS_FOLDER

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

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
            "day_of_week_Monday",
            "day_of_week_Tuesday",
            "day_of_week_Wednesday",
            "day_of_week_Thursday",
            "day_of_week_Friday",
            "day_of_week_Saturday",
            "day_of_week_Sunday",
            "events_last_24_hrs"
        ]
    ]
)
y = np.array(df_final["is_alarm"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9, random_state=101
)

svc = SVC()
svc.fit(X_train, y_train)
# y_pred = svc_linear.predict(X_test)
# print(accuracy_score(y_test,y_pred))

print(svc.score(X_train, y_train))
print(svc.score(X_test, y_test))

y_pred = svc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
