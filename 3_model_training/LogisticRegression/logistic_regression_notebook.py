# %%
import os

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from utils.timeseries_training_testing import timeseries_training_testing

output_folder = "./results"
pickle_path = '../../1_data_preparation/results/df_fin_history.pkl'
df_final = pd.read_pickle(pickle_path)

# %%
y = df_final["is_alarm"].values
df_final.drop(["is_alarm"], axis=1, inplace=True)
X = np.array(df_final)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# %%
# Tuning of model

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

# default parameters
model = LogisticRegression()
print(model.get_params())

parameters = {'penalty': ['l1', 'l2'],
              'C': [0.1, 1, 10],
              'solver': ['liblinear', 'saga']
              }
# default parameters
model = LogisticRegression()
print(model.get_params())
randm_src = RandomizedSearchCV(estimator=model, param_distributions=parameters,
                               cv=2, n_iter=10, n_jobs=-1)
randm_src.fit(X_train, y_train)

print(" Results from Random Search ")
print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)
print("\n The best score across ALL searched params:\n", randm_src.best_score_)
print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)

# %%
# tuned regular training
model_tuned_v1 = LogisticRegression(C=10, penalty='l1', solver='liblinear')

model_tuned_v1.fit(X_train, y_train)

# %%
y_pred = model_tuned_v1.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(model_tuned_v1.score(X_test, y_test))

# %%
filename = '8__logistic_regression_regular__v1'
pickle.dump(model_tuned_v2, open(f"{output_folder}/{filename}.pkl", 'wb'))

# %%
# tuned timeseries training

from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit

model_tuned_v2 = LogisticRegression(C=10, penalty='l1', solver='liblinear')

from utils.timeseries_training_testing import timeseries_training_testing

timeseries_training_testing(X, y, model_tuned_v2, 4, 0, 'is_alarm')

# %%
y_pred = model_tuned_v2.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(model_tuned_v2.score(X_test, y_test))

# %%
filename = '8__logistic_regression_timeseries__v1'
pickle.dump(model_tuned_v2, open(f"{output_folder}/{filename}.pkl", 'wb'))

