# %%
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from paths_full import DATA_PREP_RESULTS_FOLDER

pickle_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7.pkl")
df_final = pd.read_pickle(pickle_path)

# %%
y = df_final["is_alarm"].values
df_final.drop(["is_alarm"], axis=1, inplace=True)
X = np.array(df_final)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

parameters = {'learning_rate': sp_randFloat(),
              'subsample': sp_randFloat(),
              'n_estimators': sp_randInt(100, 1000),
              'max_depth': sp_randInt(4, 10)
              }

# %%
# default parameters
model = GradientBoostingClassifier()
print(model.get_params())

# %%
randm_src = RandomizedSearchCV(estimator=model, param_distributions=parameters,
                               cv=2, n_iter=10, n_jobs=-1)
randm_src.fit(X_train, y_train)

print(" Results from Random Search ")
print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)
print("\n The best score across ALL searched params:\n", randm_src.best_score_)
print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)
# Results from Random Search
#
#  The best estimator across ALL searched params:
#  GradientBoostingClassifier(learning_rate=0.8701215345496025, max_depth=6,
#                            n_estimators=152, subsample=0.539914860633278)
#
#  The best score across ALL searched params:
#  0.875626486844576
#
#  The best parameters across ALL searched params:
#  {'learning_rate': 0.8701215345496025, 'max_depth': 6, 'n_estimators': 152, 'subsample': 0.539914860633278}

# %%
# tuned
model_tuned = GradientBoostingClassifier(learning_rate=0.8701215345496025, max_depth=6, n_estimators=152, subsample=0.539914860633278)
model_tuned.fit(X_train, y_train)
y_pred = model_tuned.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(model_tuned.score(X_train, y_train))
print(model_tuned.score(X_test, y_test))
# [[27097  1610]
#  [ 2409  7343]]
# 0.9332596613254461
# 0.895499102940794

# %%
# default
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
# [[27403  1304]
#  [ 3656  6096]]
# 0.8724932557610426
# 0.8710314880782132

# %%
# get importance
importance = model_tuned.feature_importances_
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# %%
feat_importances = pd.Series(model_tuned.feature_importances_, index=df_final.columns)
feat_importances.nlargest(20).plot(kind='barh')
pyplot.title("Top 20 important features")
pyplot.show()

# %%
# check columns index
df_final.iloc[0:5, 30:35]

# %%
