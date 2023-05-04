import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from isw_text_preprocessing import (do_preprocessing)

df = pd.DataFrame(columns=["Name", "Date", "Text"])

df_list = []

print("Reading folder contents")

directory = './data/isw/'
files = os.listdir(directory)

for filename in files:
    if filename.startswith('assessment-'):
        # Open the file
        with open(os.path.join(directory, filename), 'r', encoding="utf-8") as file:
            # Do something with the file
            name = filename.split(".")[0]
            date = filename.replace("assessment-", "").replace(".txt", "")
            text = file.read()
            row_df = pd.DataFrame({"Name": [name], "Date": [date], "Text": [text]})
            df_list.append(row_df)
        break

df = pd.concat(df_list, ignore_index=True)

print("Successfully read the input data")

print("Find tokens")
filteredDf = df["Text"].apply(lambda d: " ".join(do_preprocessing(d)))

# To be uncommented if you want to see the most common words
# print("Find most common words")
# all_words = []
# for tokens in filteredDf:
#     for word in tokens.split(" "):
#         all_words.append(word)
# all_words = nltk.FreqDist(all_words)
# print("Top 30 frequenty used words: ")
# print(all_words.most_common(30))

frequent_words = {
    "russian",
    "force",
    "forces",
    "ukrainian",
    "ukraine",
    "oblast",
    "military",
    "reported",
    "effort",
    "likely",
    "claimed",
    "russia",
    "area",
    "operation",
    "continued",
    "city",
    "general",
    "near",
    "attack",
    "official",
    "staff",
    "also",
    "stated",
    "source",
    "oblast",
    "pm",
    "am",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

filteredDf = filteredDf.apply(
    lambda d: " ".join(w for w in d.split() if w not in frequent_words)
)
df["Tokens"] = filteredDf

filenames = df["Name"]
dates = df["Date"]

print("Create vectors")
tfidf = TfidfVectorizer(smooth_idf=True, use_idf=True, max_features=20)
vectors = tfidf.fit_transform(df["Tokens"])

feature_names = tfidf.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
dictionaries = df.to_dict(orient="records")


res = __builtins__.zip(filenames, dates, dictionaries)
res_df = pd.DataFrame(res, columns=["Name", "Date", "Keywords"])
res_df["Keywords"] = res_df["Keywords"].apply(
    lambda d: {k: v for k, v in d.items() if v > 0}
)
res_df["Keywords"] = res_df["Keywords"].apply(
    lambda d: dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
)

res_df.to_csv("./data_preparators/results/tfidf-new.csv", index=False)
print("Successfully written to .csv")
