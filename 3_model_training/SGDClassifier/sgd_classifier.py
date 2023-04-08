import numpy as np
import pandas as pd
import os

from paths_full import DATA_PREP_RESULTS_FOLDER


from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier


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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=101
)


sgd = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)

# Evaluate the model
accuracy = sgd.score(X_test, y_test)
print(accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict the labels for the test set
y_pred = sgd.predict(X_test)

# Calculate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=sgd.classes_
)
disp.plot()