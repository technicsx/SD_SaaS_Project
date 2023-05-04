# %%
# !pip install numpy
# !pip install pandas
# !pip install scikit-learn
# !pip install matplotlib

# %%
import numpy as np
import pandas as pd
import pickle
import os


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import pyplot

from sklearn.model_selection import GridSearchCV

pickle_path = '../data_preparators/results/df_fin_history_no_isw.pkl'
df_final = pd.read_pickle(pickle_path)

# %%



# %%
df_final.head(2)

# %%
# df_new = df_new[['region_id','day_datetimeEpoch', 'is_alarm']]

# %%
df_final.head(5)

# %%
df_final = df_final.sort_index(axis=1)

# %%
df_final.describe()

# %%
y = df_final["is_alarm"].values
df_final.drop(["is_alarm"], axis=1, inplace=True)
X = np.array(df_final)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# %%
non_zero_alarm = np.count_nonzero(y)
zero_alarm = y.size - non_zero_alarm
print(non_zero_alarm)
print(zero_alarm)

# %%
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# %%
# default parameters
# rf = RandomForestClassifier()
# print(rf.get_params())

# %%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
# rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
# rf_random.fit(X_train, y_train)

# %%
print(" Results from Random Search ")
print("\n The best estimator across ALL searched params:\n", rf_random.best_estimator_)
print("\n The best score across ALL searched params:\n", rf_random.best_score_)
print("\n The best parameters across ALL searched params:\n", rf_random.best_params_)
# Results from Random Search
#
#  The best estimator across ALL searched params:
#  RandomForestClassifier(bootstrap=False, max_depth=90, min_samples_leaf=4,
#                        min_samples_split=10, n_estimators=600)
#
#  The best score across ALL searched params:
#  0.9038515393268926
#
#  The best parameters across ALL searched params:
#  {'n_estimators': 600, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 90, 'bootstrap': False}

# %%
# rf_grid = GridSearchCV(estimator=rf, param_grid=random_grid, cv=5)
# rf_grid.fit(X_train, y_train)

# print(" Results from Grid Search ")
# print(f"Best hyperparameters: {grid_search.best_params_}")
# print(f"Best cross-validation score: {grid_search.best_score_}")

# %%
# tuned
model_tuned = RandomForestClassifier(bootstrap=False, max_depth=90, min_samples_leaf=4,
                       min_samples_split=10, n_estimators=600)
model_tuned.fit(X_train, y_train)

print(model_tuned.score(X_test, y_test))

# Predict the labels for the test set
y_pred = model_tuned.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=model_tuned.classes_
)
disp.plot()
# 0.9133622819106061
# [[27899   808]
#  [ 2524  7228]]

# city update
# 0.9148183780129489
# [[27858   849]
#  [ 2427  7325]]

# %%
# default
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=clf.classes_
)
disp.plot()
# 0.908317948984633
# [[27859   848]
#  [ 2678  7074]]

# %%
# get importance
importance = model_tuned.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# %%
feat_importances = pd.Series(model_tuned.feature_importances_, index=df_final.columns)
feat_importances.nlargest(20).plot(kind='barh')
pyplot.title("Top 20 important features")
pyplot.show()

# %%
output_folder = "./results"
filename = '8__random_forrest__v1'
pickle.dump(model_tuned, open(f"{output_folder}/{filename}.pkl", 'wb'))


# %%
pickle_path = '../data_preparators/results/df_fin_new.pkl'
df_new = pd.read_pickle(pickle_path)

# %%
df_new = df_new.sort_index(axis=1)

X = np.array(df_new)

# %% [markdown]
#

# %%
df_new.describe()

# %%
diff_cols = set(df_final.columns) ^ set(df_new.columns)

# Print the column differences
print(f"Column differences: {diff_cols}")

if len(diff_cols) == 0:
    print("The two dataframes have the same columns")
else:
    print("The two dataframes have different columns")

# %%
X

# %%
with open(f"{output_folder}/{filename}.pkl", 'rb') as file:
    model = pickle.load(file)

y_pred = model.predict(X)

# %%
y_pred

# %%
y_pred.shape

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %%
# Evaluate the performance of the model
accuracy = accuracy_score(np.zeros(X.shape[0]), y_pred)
precision = precision_score(np.zeros(X.shape[0]), y_pred, zero_division=0)
recall = recall_score(np.zeros(X.shape[0]), y_pred, zero_division=0)
f1 = f1_score(np.zeros(X.shape[0]), y_pred, zero_division=0)

# Print the evaluation metrics
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 score: {:.2f}'.format(f1))


# %%
df_new['is_alarm'] = y_pred
df_new.head(1)

# %%
df_fin.to_pickle("results/df_fin_predicted.pkl")

