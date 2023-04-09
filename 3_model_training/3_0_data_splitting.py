# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from paths_full import DATA_PREP_RESULTS_FOLDER
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

csv_path = os.path.join(DATA_PREP_RESULTS_FOLDER, "df_weather_v7_v3.pkl")
df_final = pd.read_pickle(csv_path)


# %%



# %%

# %%
df_final.shape

# %%

# %%



# %%

# Initialize the time-series cross-validation object
cv = TimeSeriesSplit(n_splits=2)

X = df_final.drop(['is_alarm'], axis=1)
y = df_final['is_alarm']

logreg_liblinear = LogisticRegression(penalty='l1',solver='liblinear')

# Train and test the model on each fold of the cross-validation object
for train_index, test_index in cv.split(X):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Train the model on the training set
    logreg_liblinear.fit(X_train, y_train)
    print("Accuracy WITHOUT feature selection:", logreg_liblinear.score(X_test, y_test))


# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_pred = logreg_liblinear.predict(X_test)

# Calculate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                             display_labels=logreg_liblinear.classes_)
disp.plot()


# %%
