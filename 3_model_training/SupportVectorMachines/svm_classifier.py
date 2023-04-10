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
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

csv_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7_v3.pkl")
df_final = pd.read_pickle(csv_path)

# %%
df_final.shape

# %%
# df_final.shape

# %%
# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    df_final.drop(["is_alarm"], axis=1).values,
    df_final["is_alarm"].values,
    test_size=0.25,
    random_state=101,
)

# %%
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                             display_labels=clf.classes_)
disp.plot()

# %%

# logreg_liblinear = LogisticRegression(
#     solver="liblinear", class_weight="balanced", random_state=0
# )

# logreg_liblinear.fit(X_train, y_train)
pred_test = clf.predict(X_test)
print(pred_test)

print(clf.classes_)
print(clf.intercept_)
print(clf.coef_)

print(clf.score(X_test, y_test))
print(clf.score(X_train, y_train))


y_pred = clf.predict(X_test)
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

