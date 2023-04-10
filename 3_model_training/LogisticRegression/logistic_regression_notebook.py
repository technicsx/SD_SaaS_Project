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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from paths_full import *
from sklearn.feature_selection import SelectKBest, f_classif
from utils.model_confusion_matrix_out import *
from utils.model_features_info_out import *
from utils.timeseries_training_testing import *

# %%
csv_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7_v3.pkl")
df_final = pd.read_pickle(csv_path)

# %%
# Splitting the data into ordinary 75/25 training and testing data
X_train_ordinary, X_test_ordinary, y_train_ordinary, y_test_ordinary = train_test_split(
    df_final.drop(["is_alarm"], axis=1).values,
    df_final["is_alarm"].values,
    test_size=0.25,
    random_state=101,
)

# %%
# selector = SelectKBest(f_classif, k=10)
# X_train_new = selector.fit_transform(X_train_ordinary, y_train_ordinary)
# X_test_new = selector.transform(X_test_ordinary)

# %%
# Fit a logistic regression model with L1 regularization
logistic_regression_1 = LogisticRegression(penalty='l1',solver='liblinear')
logistic_regression_1.fit(X_train_ordinary, y_train_ordinary)

print("Accuracy 75/25 train_test_split:", logistic_regression_1.score(X_test_ordinary, y_test_ordinary))

importance_1 = logistic_regression_1.coef_[0]
y_pred_1 = logistic_regression_1.predict(X_test_ordinary)

# %%
output_model_confusion_matrix(logistic_regression_1, y_test_ordinary, y_pred_1)

# %%
output_overall_features_importance_diagram(importance_1)

# %%
# Fit a logistic regression model with L1 regularization
logistic_regression_2 = LogisticRegression(penalty='l1',solver='liblinear')
timeseries_training_testing(df_final,logistic_regression_2, 2)
importance_2 = logistic_regression_2.coef_[0]

# %%
y_pred_2 = logistic_regression_2.predict(X_test_ordinary)

# %%
output_model_confusion_matrix(logistic_regression_2, y_test_ordinary, y_pred_2)

# %% [markdown]
#
