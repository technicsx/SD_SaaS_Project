# %%
import numpy as np
import pandas as pd
import os

# %%
from paths_full import DATA_PREP_RESULTS_FOLDER


# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier


# %%
csv_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7_v3.pkl")
df_final = pd.read_pickle(csv_path)

# %%
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix

# %%
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from utils.timeseries_training_testing import *

# %%
# define model
model = SGDClassifier()

X = df_final.drop(['is_alarm'], axis=1)
y = df_final['is_alarm']

# define parameter distribution for search
param_dist = {
    'alpha': uniform(0, 1),
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': uniform(0, 1)
}

# define evaluation method
cv = 5

# define search
search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=1, scoring='accuracy', n_jobs=-1, cv=cv)

# execute search
search.fit(X, y)

# %%
print(" Results from Random Search ")
print("\n The best estimator across ALL searched params:\n", search.best_estimator_)
print("\n The best score across ALL searched params:\n", search.best_score_)
print("\n The best parameters across ALL searched params:\n", search.best_params_)

# %%
X_train_ordinary, X_test_ordinary, y_train_ordinary, y_test_ordinary = train_test_split(
    df_final.drop(["is_alarm"], axis=1).values,
    df_final["is_alarm"].values,
    test_size=0.25,
    random_state=101,
)

# %%
# Feature selection
selector = SelectKBest(f_classif, k=2)
X_train_new = selector.fit_transform(X_train_ordinary, y_train_ordinary)
X_test_new = selector.transform(X_test_ordinary)

# %%
# Fit a logistic regression model with L1 regularization
sgd = SGDClassifier(penalty='l1')
sgd.fit(X_train_new, y_train_ordinary)

print("Accuracy 75/25 train_test_split:", sgd.score(X_test_new, y_test_ordinary))

importance = sgd.coef_[0]
y_pred = sgd.predict(X_test_new)

# %%
y_pred = sgd.predict(X_test_new)

# %%
confusion_matrix = confusion_matrix(y_test_ordinary, y_pred)
print(confusion_matrix)

# %%
output_model_confusion_matrix(sgd, y_test_ordinary, y_pred)

# %%
sgd_2 = SGDClassifier(penalty='l1')
timeseries_training_testing(df_final,sgd_2, 2)
importance_2 = sgd_2.coef_[0]

y_pred_2 = sgd_2.predict(X_test_ordinary)


# %%
confusion_matrix = confusion_matrix(y_test_ordinary, y_pred_2)
print(confusion_matrix)


# %%
output_model_confusion_matrix(sgd_2, y_test_ordinary, y_pred_2)


# %%
