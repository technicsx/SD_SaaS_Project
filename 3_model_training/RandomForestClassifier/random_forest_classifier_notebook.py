# %%
import numpy as np
import pandas as pd
import os

from paths_full import DATA_PREP_RESULTS_FOLDER

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot


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
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Calculate the confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=clf.classes_
)
disp.plot()

# %%
# get importance
importance = clf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

# %%
feat_importances = pd.Series(clf.feature_importances_, index=df_final.columns)
feat_importances.nlargest(20).plot(kind='barh')
pyplot.title("Top 20 important features")
pyplot.show()

# %%
