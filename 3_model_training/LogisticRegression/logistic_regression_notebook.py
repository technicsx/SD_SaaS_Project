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
from matplotlib import pyplot

csv_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7.pkl")
df_final = pd.read_pickle(csv_path)

# %%
df_final.shape

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
model = LogisticRegression(
    solver="liblinear", class_weight="balanced", random_state=0
)

model.fit(X_train, y_train)
pred_test = model.predict(X_test)
print(pred_test)

print(model.classes_)
print(model.intercept_)
print(model.coef_)

print(model.score(X_test, y_test))
print(model.score(X_train, y_train))


y_pred = model.predict(X_test)
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

# %%
# get importance
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# %%
feat_importances = pd.Series(importance, index=df_final.drop(["is_alarm"], axis=1).columns)
feat_importances.nlargest(20).plot(kind='barh')
pyplot.title("Top 20 important features")
pyplot.show()

# %%
feat_importances.nsmallest(20).plot(kind='barh')
pyplot.title("Top 20 negative features")
pyplot.show()

# %%
