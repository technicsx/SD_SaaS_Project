# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from paths_full import DATA_PREP_RESULTS_FOLDER

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

csv_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_fin.pkl")
df_final = pd.read_pickle(csv_path)

# %%
df_final.shape

# %%
# Fill NaN values
# df_final[["event_all_region"]] = df_final[["event_all_region"]].fillna(value=0)

# %%
# df_final.shape

# %%
# X = np.array(
#     df_final[
#         [
#             "region_id",
#             "event_all_region",
#             "day_datetimeEpoch",
#             "hour_datetimeEpoch",
#             "ukrainian_holiday",
#             "russian_holiday",
#             "hour_temp",
#             "hour_snow",
#             "hour_visibility",
#             "hour_conditions_code",
#             "lunar_eclipse",
#             "solar_eclipse",
#             "moonphased_eclipse",
#             "alarmed_regions_count",
#         ]
#     ]
# )
# df_final except is_alarm
# X = np.array(df_final.drop(["is_alarm"], axis=1))

# y = np.array(df_final["is_alarm"])

# %%
# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df_final.drop(["is_alarm"], axis=1).values,
    df_final["is_alarm"].values,
    test_size=0.25,
    random_state=101,
)

# %%
regr = LinearRegression()
regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))

# %%

logreg_liblinear = LogisticRegression(
    solver="liblinear", class_weight="balanced", random_state=0
)

logreg_liblinear.fit(X_train, y_train)
pred_test = logreg_liblinear.predict(X_test)
print(pred_test)

print(logreg_liblinear.classes_)
print(logreg_liblinear.intercept_)
print(logreg_liblinear.coef_)

print(logreg_liblinear.score(X_test, y_test))
print(logreg_liblinear.score(X_train, y_train))


y_pred = logreg_liblinear.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

# %%
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted 0s", "Predicted 1s"))
ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual 0s", "Actual 1s"))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
plt.show()
