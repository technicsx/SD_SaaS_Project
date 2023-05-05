# %%
import os

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from utils.timeseries_training_testing import timeseries_training_testing

output_folder = "../results"
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
model = GradientBoostingClassifier()
print(model.get_params())

parameters = {'learning_rate': sp_randFloat(),
              'subsample': sp_randFloat(),
              'n_estimators': sp_randInt(100, 1000),
              'max_depth': sp_randInt(4, 10)
              }
# default parameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
# %%
print(" Results from Random Search ")
print("\n The best estimator across ALL searched params:\n", rf_random.best_estimator_)
print("\n The best score across ALL searched params:\n", rf_random.best_score_)
print("\n The best parameters across ALL searched params:\n", rf_random.best_params_)
# %%
# tuned regular training
model_tuned_v1 = RandomForestClassifier(bootstrap=False, max_depth=90, min_samples_leaf=4,
                       min_samples_split=10, n_estimators=600)
model_tuned_v1.fit(X_train, y_train)

# %%
y_pred = model_tuned_v1.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(model_tuned_v1.score(X_test, y_test))

# %%
# get importance
importance = model_tuned_v1.feature_importances_
from utils.model_features_info_out import output_overall_features_importance_diagram
output_overall_features_importance_diagram(importance)

# %%
feat_importances = pd.Series(model_tuned_v1.feature_importances_, index=df_final.columns)
feat_importances.nlargest(10).plot(kind='barh')
pyplot.title("Top 10 important features")
pyplot.show()

# %%
filename = '8__random_forest_regular__v1'
pickle.dump(model_tuned_v1, open(f"{output_folder}/{filename}.pkl", 'wb'))

# %%
# tuned timeseries training

from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit

model_tuned_v2 = RandomForestClassifier(bootstrap=False, max_depth=90, min_samples_leaf=4,
                       min_samples_split=10, n_estimators=600)

from utils.timeseries_training_testing import timeseries_training_testing

timeseries_training_testing(X, y, model_tuned_v2, 4, 0, 'is_alarm')

# %%
y_pred = model_tuned_v2.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(model_tuned_v2.score(X_test, y_test))

# %%
# get importance
importance = model_tuned.feature_importances_
from utils.model_features_info_out import output_overall_features_importance_diagram
output_overall_features_importance_diagram(importance)

# %%
feat_importances = pd.Series(model_tuned.feature_importances_, index=df_final.columns)
feat_importances.nlargest(10).plot(kind='barh')
pyplot.title("Top 10 important features")
pyplot.show()

# %%
filename = '8__random_forest_timeseries__v1'
pickle.dump(model_tuned_v2, open(f"{output_folder}/{filename}.pkl", 'wb'))

