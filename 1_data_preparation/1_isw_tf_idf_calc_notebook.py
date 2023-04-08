# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] cell_id="a78848a5d0fd4e34a5897a47dc230dcc" deepnote_app_coordinates={"h": 5, "w": 12, "x": 0, "y": 1} deepnote_cell_type="text-cell-h1" formattedRanges=[{"fromCodePoint": 0, "marks": {"bold": true}, "toCodePoint": 17, "type": "marks"}] is_collapsed=false
# # ISW preprocessing

# %% [markdown] cell_id="7133bbff0c7644cdac4b3394e5327dd8" deepnote_app_coordinates={"h": 5, "w": 12, "x": 0, "y": 31} deepnote_cell_type="text-cell-h3" formattedRanges=[] is_collapsed=false
# ### Import and download all dependecies

# %% cell_id="8e138bae7b244bdea0d767dd645b62d8" deepnote_app_coordinates={"h": 25, "w": 12, "x": 0, "y": 37} deepnote_app_is_output_hidden=true deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=2920 execution_start=1680019070283 source_hash="ec535853"
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

from text_preprocessing import (do_preprocessing)

# %% [markdown] cell_id="3adde034474b4e01b133850f0598de42" deepnote_app_coordinates={"h": 5, "w": 12, "x": 0, "y": 117} deepnote_cell_type="text-cell-h3" formattedRanges=[] is_collapsed=false
# ### Reading target files

# %% cell_id="9b0dd02d4a1d46c9910d60f25e24d79f" deepnote_app_coordinates={"h": 13, "w": 12, "x": 0, "y": 123} deepnote_app_is_output_hidden=false deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=928 execution_start=1680019073228 source_hash="2e7a14d0"
from paths_full import *

df = pd.DataFrame(columns=["Name", "Date", "Text"])

df_list = []

print("Reading folder contents")
for root, dirs, files in os.walk(ISW_SCRAPPING_FOLDER):
    for filename in files:
        if filename.endswith(".txt"):
            with open(os.path.join(root, filename), encoding="utf-8") as file:
                name = filename.split(".")[0]
                date = filename.replace("assessment-", "").replace(".txt", "")
                text = file.read()
                row_df = pd.DataFrame({"Name": [name], "Date": [date], "Text": [text]})
                df_list.append(row_df)
df = pd.concat(df_list, ignore_index=True)
print("Successfully read the input data")

# %% [markdown] cell_id="17b1138cf7c9457aa477a30ccadacbf0" deepnote_app_coordinates={"h": 5, "w": 12, "x": 0, "y": 137} deepnote_cell_type="text-cell-h3" formattedRanges=[] is_collapsed=false
# ### TF-IDF creation

# %% cell_id="09538d60d2d54f25bc752ee441583c16" deepnote_app_coordinates={"h": 44, "w": 12, "x": 0, "y": 143} deepnote_cell_type="code" deepnote_table_loading=false deepnote_table_state={"filters": [], "pageIndex": 0, "pageSize": 10, "sortBy": []} deepnote_to_be_reexecuted=false execution_millis=95017 execution_start=1680019074182 source_hash="33982a8a"
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
tfidf = TfidfVectorizer(smooth_idf=True, use_idf=True)
vectors = tfidf.fit_transform(df["Tokens"])

# store content
with open("results/tfidf.pkl", "wb") as handle:
    pickle.dump(tfidf, handle)


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

res_df.to_csv("results/tfidf.csv", index=False)
print("Successfully written to .csv")
