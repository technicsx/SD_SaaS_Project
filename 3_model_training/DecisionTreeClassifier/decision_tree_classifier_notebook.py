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
import numpy as np
import pandas as pd
import pickle
import os

from paths_full import DATA_PREP_RESULTS_FOLDER

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV


pickle_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7_city.pkl")
df_final = pd.read_pickle(pickle_path)

# %%
y = df_final["is_alarm"].values
df_final.drop(["is_alarm"], axis=1, inplace=True)
X = np.array(df_final)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# %%
# decision tree classifier for classification in scikit-learn
# define model
model = DecisionTreeClassifier()
print(model.get_params())

# %%
# define parameter grid for search
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'criterion': ['gini', 'entropy']
}

# define evaluation method
cv = 5

# define search
search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=cv)

# execute search
search.fit(X_train, y_train)

# summarize result
print('Best Score: %s' % search.best_score_)
print('Best Hyperparameters: %s' % search.best_params_)

# %%
print(" Results from GridSearchCV ")
print("\n The best estimator across ALL searched params:\n", search.best_estimator_)
print("\n The best score across ALL searched params:\n", search.best_score_)
print("\n The best parameters across ALL searched params:\n", search.best_params_)
# Results from GridSearchCV
#
#  The best estimator across ALL searched params:
#  DecisionTreeClassifier(max_depth=7)
#
#  The best score across ALL searched params:
#  0.8673123801475608
#
#  The best parameters across ALL searched params:
#  {'criterion': 'gini', 'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 2}

# %%
# tuned
model_tuned = DecisionTreeClassifier(max_depth=7)
model_tuned.fit(X_train, y_train)
print(model_tuned.score(X_test, y_test))
# 0.8670792272289971

# %%
# default
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
# 0.8878806001196079

# %%
# Predict the labels for the test set
y_pred = model.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=model.classes_
)
disp.plot()

# %%
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# %%
feat_importances = pd.Series(model.feature_importances_, index=df_final.columns)
feat_importances.nlargest(20).plot(kind='barh')
pyplot.title("Top 20 important features")
pyplot.show()

# %%
# save the model to disk
output_folder = "../results"
filename = '8__decision_tree__v1'
pickle.dump(model_tuned, open(f"{output_folder}/{filename}.pkl", 'wb'))
