# %%
import os

import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

pickle_path = '../data_preparators/results/df_fin_history.pkl'
df_final = pd.read_pickle(pickle_path)


# %%
y = df_final["is_alarm"].values
df_final.drop(["is_alarm"], axis=1, inplace=True)
X = np.array(df_final)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# %%
df_final.head(50)

# %%

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
y = df_final["is_alarm"].values
df_final.drop(["is_alarm"], axis=1, inplace=True)
X = np.array(df_final)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# %%

from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import chi2

# Select the top 5 features based on the F-test score
selector = SelectKBest(score_func=chi2, k=8)
X_new = selector.fit_transform(X, y)

# %%
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=4)
X = df_final.drop(['is_alarm'], axis=1)
y = df_final['is_alarm']

features = X.columns

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=df_final.drop(['is_alarm'], axis=1).columns)

# Select the top 5 features based on the F-test score
# selector = SelectKBest(score_func=chi2, k=20)
# X = selector.fit_transform(X, y)
# selected_features = features[selector.get_support()]
# X = pd.DataFrame(X, columns=selected_features)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
# clf = ExtraTreesClassifier(n_estimators=50)
#clf.fit(X, y)
## Select the top 5 features
#selector = SelectFromModel(clf, prefit=True, max_features=20)
#selected_features = features[selector.get_support()]
#X = selector.transform(X)
#X = pd.DataFrame(X, columns=selected_features)


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
# default parameters
model = GradientBoostingClassifier()
print(model.get_params())
randm_src = RandomizedSearchCV(estimator=model, param_distributions=parameters,
                               cv=2, n_iter=10, n_jobs=-1)
randm_src.fit(X_train, y_train)

print(" Results from Random Search ")
print("\n The best estimator across ALL searched params:\n", randm_src.best_estimator_)
print("\n The best score across ALL searched params:\n", randm_src.best_score_)
print("\n The best parameters across ALL searched params:\n", randm_src.best_params_)
# Results from Random Search
# Results from Random Search

# The best estimator across ALL searched params:
# GradientBoostingClassifier(learning_rate=0.26455766539870096, max_depth=9,
                           n_estimators=537, subsample=0.7541331375277148)

# The best score across ALL searched params:
# 0.9074072892217155

# The best parameters across ALL searched params:
# {'learning_rate': 0.26455766539870096, 'max_depth': 9, 'n_estimators': 537, 'subsample': 0.7541331375277148}

# %%



model_tuned = GradientBoostingClassifier(learning_rate=0.26455766539870096, max_depth=9, n_estimators=537, subsample=0.7541331375277148)

for train_index, val_index in tscv.split(X):
    # Split the data into training and validation sets
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Train a gradient boosting model using the selected features

    model_tuned.fit(X_train_fold, y_train_fold)

    # Evaluate the model on the validation set
    score = model_tuned.score(X_val_fold, y_val_fold)
    print(f'Validation set score: {score:.2f}')

# %%
# tuned
model_tuned = GradientBoostingClassifier(learning_rate=0.4361416688248827, max_depth=5,
                           n_estimators=589, subsample=0.6657227329092461)
model_tuned.fit(X_train, y_train)

# %%
y_pred = model_tuned.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(model_tuned.score(X_test, y_test))

# %%
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


clf = ExtraTreesClassifier(n_estimators=50)
clf.fit(X, y)

# Select the top 5 features
selector = SelectFromModel(clf, prefit=True, max_features=10)
X_new = selector.transform(X)

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
feat_importances.nlargest(10).plot(kind='barh')
pyplot.title("Top 10 important features")
pyplot.show()

# %%
# save the model to disk
output_folder = "./results"
filename = '8__gradient_boosting__v8'

# %%
# save the model to disk
pickle.dump(model_tuned, open(f"{output_folder}/{filename}.pkl", 'wb'))

# %%
# load the model from disk
loaded_model = pickle.load(open(f"{output_folder}/{filename}.pkl", 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# %%
pickle_path = '../data_preparators/results/df_fin_new.pkl'
df_new = pd.read_pickle(pickle_path)
df_new = df_new.sort_index(axis=1)
X_df_new = np.array(df_new)

# %%
with open(f"{output_folder}/{filename}.pkl", 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X_df_new)
y_pred


# %%
y_pred


# %%
