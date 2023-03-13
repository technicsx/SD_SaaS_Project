# ISW preprocessing

[assessment-march-3 (2).html][nextjournal#file#022cd523-1d51-42ac-860a-7c210c9832a9]

```bash id=5ce2b639-b88c-4f3b-b087-928894f0effe
# Installs
pip install nltk num2words
pip install -U scikit-learn
```

```python id=a21f50e2-b9c6-43d7-8101-c0a47d6613ba
# All import
import pandas as pd
import numpy as np
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from num2words import num2words
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import string

# Downloads
nltk.download('word_tokenize')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# 
```

```python id=c5e844d2-bd49-44e3-8a11-c0248e58549f
# Functions
def to_lower_case(text):
  return "".join([i.lower() for i in text])

stop_punctuation = string.punctuation
def remove_punctuation(text):
  return "".join([i for i in text if i not in stop_punctuation])

def remove_long_dash(text):
  return re.sub(r'—', ' ', text)

def remove_urls(text):
  return re.sub(r'http\S+', '', text)

def remove_one_letter_words(tokens):
  return list(filter(lambda token: len(token) > 1, tokens))

def tokenize_text(text):
  return nltk.tokenize.word_tokenize(text)

stop_words = set(nltk.corpus.stopwords.words('english'))
avoid_stop_words = set(["not","n't","no"])
stop_words = stop_words - avoid_stop_words

def remove_stop_words(tokens):
  return [i for i in tokens if i not in stop_words]

def do_stemming(tokens):
  ps = nltk.PorterStemmer()
  return [ps.stem(word) for word in tokens]

def do_lemmatization(tokens):
  wn = nltk.WordNetLemmatizer()
  return [wn.lemmatize(word) for word in tokens]

def remove_numeric_words(text):
  return re.sub(r'\S*\d+\S*', '', text)

def convert_nums_to_words(data):
  tokens = data
  new_text = []
  for word in tokens:
    if word.isdigit():
      if int(word)<1000000000:
        word = num2words(word)
      else: 
        word = ""
    new_text.extend(tokenize_text(re.sub("(-|,\s?)|\s+", " ", word)))
  return new_text

def do_preprocessing(data):
  text_clean = data
  text_clean = remove_urls(text_clean)
  text_clean = remove_punctuation(text_clean)
  text_clean = remove_long_dash(text_clean)
  text_clean = to_lower_case(text_clean)
  text_clean = remove_numeric_words(text_clean)
  words = tokenize_text(text_clean)
  words = remove_one_letter_words(words)
  words = remove_stop_words(words)
  lemmatized = do_lemmatization(words)
  res = convert_nums_to_words(lemmatized)
  return res

# test_input = """At a a http://hello a a eight b n f s a s w q e z o'clock on Thursday morning
# ... Arthur didn't feel 20th a very a good a goodness."""

# result = do_preprocessing(test_input)
# print(result)

# f = open(_, "r").read()
# print(do_preprocessing(f))
```

# Vectorizer Test

```python id=6ccf655d-64b1-4c2d-86f0-6ac76bc04c48
documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([documentA, documentB])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
df
```

[result][nextjournal#output#6ccf655d-64b1-4c2d-86f0-6ac76bc04c48#result]

[assessment-december-3.html][nextjournal#file#3bbf1eaa-c623-46ef-ba52-2b1bb8150636]

[assessment-december-31.html][nextjournal#file#6e9bae87-7207-42e9-957c-638e2e53cb5e]

[assessment-january-31-2023.html][nextjournal#file#cceae52d-12b1-4fd2-9707-5f974bda6ffc]

# Do Vectorizing

```python id=23d49574-c621-485a-9e75-1eb38413222a
from pathlib import Path
import os
# files = ["_", "_", "_", "_"]

# docs = map(lambda x: open(x).read(), files)

files = [open([reference][nextjournal#reference#242eafcf-3c9c-49e5-8fb5-f5faf0634d17]), open([reference][nextjournal#reference#1a0bf92b-b557-4cb0-ae9e-469664e73456]), open([reference][nextjournal#reference#b6a07561-ce98-440d-84bc-a3198d909ff0]), open([reference][nextjournal#reference#dd1bff04-90f9-45e1-96d9-6e43f3743698])]

docs = list(map(lambda d: d.read(), files))

df = pd.DataFrame(docs, columns = ["Text"])
file_names = list(map(lambda d: os.path.realpath(d.name).rsplit('/', 1)[1].split('.')[0], files))

df["Tokens"] = df["Text"].apply(lambda d: " ".join(do_preprocessing(d)))

tfidf = TfidfVectorizer(smooth_idf=True,use_idf=True)
vectors = tfidf.fit_transform(df["Tokens"])
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
# print(df)
dictionaries = df.to_dict(orient='records')
# print(dictionaries)
res = __builtins__.zip(file_names, dictionaries)
# print(list(res))
res_df = pd.DataFrame(res, columns=["Date","Keywords"])
res_df
# df["Date"] = np.resize(file_names,len(df))
# print(res_df["Date"].tolist())

# res_df

# res_df.to_csv('/results/out.csv', index=False)  

# print(open([reference][nextjournal#reference#525e9957-f106-4b93-a5cc-fa7ff73c811c]).read())
```

[out.zip][nextjournal#file#696ec1c6-2f92-4000-b7fe-069b4da30d8b]

## Zipping and Vectorizing

```python id=ddcbe78e-ee11-4235-a250-c90d9dfe32d2
# importing required modules
from zipfile import ZipFile, ZipInfo
from pathlib import Path
import os

# specifying the zip file name
file_name = [reference][nextjournal#reference#f0c20232-9d5e-4791-b966-2976173dc4aa]
# file_name = [reference][nextjournal#reference#c437612b-f6d7-4f2f-b745-54a7d4487a9c]

df = pd.DataFrame(columns = ["Name", "Date", "Text"])
print("Read zip")
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zipfile:
      for file in zipfile.infolist():
        if not ZipInfo.is_dir(file):
          filename = file.filename.rsplit('/', 1)[1].split('.')[0]
          date = filename.replace("assessment-", "")
          text = zipfile.read(file.filename).decode('utf-8')
          df = df.append({"Name": filename, "Date": date, "Text": text}, ignore_index = True)

print("Find tokens")          
df["Tokens"] = df["Text"].apply(lambda d: " ".join(do_preprocessing(d)))

filenames = df["Name"]
dates = df["Date"]

print("Make vectors")
tfidf = TfidfVectorizer(smooth_idf=True,use_idf=True)
vectors = tfidf.fit_transform(df["Tokens"])

# store content
with open("/results/tfidf.pkl", "wb") as handle:
  pickle.dump(tfidf, handle)

feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
dictionaries = df.to_dict(orient='records')

print("Into result")
res = __builtins__.zip(filenames, dates, dictionaries)
res_df = pd.DataFrame(res, columns=["Name","Date","Keywords"])
res_df["Keywords"] = res_df["Keywords"].apply(lambda d: {k: v for k, v in d.items() if v > 0})
res_df
```

[result][nextjournal#output#ddcbe78e-ee11-4235-a250-c90d9dfe32d2#result]

[tfidf.pkl][nextjournal#output#ddcbe78e-ee11-4235-a250-c90d9dfe32d2#tfidf.pkl]

```python id=f54371ae-8a6d-40d6-9e3a-f0469cb726b5
filename = "out_no_zero"
compression_options = dict(method='zip', archive_name=f'{filename}.csv')
res_df.to_csv(f'/results/{filename}.zip', compression=compression_options, index=False)
# res_df.to_csv('/results/out.csv', index=False)
```

[out_no_zero.zip][nextjournal#output#f54371ae-8a6d-40d6-9e3a-f0469cb726b5#out_no_zero.zip]

[test.zip][nextjournal#file#12ab85bc-e83e-4444-99d6-6750590b3557]


[nextjournal#file#022cd523-1d51-42ac-860a-7c210c9832a9]:
<https://nextjournal.com/data/Qmbf6WrCX2rcEk965o69spXiTMyScd7fz5dLwNsQnyDeWt?content-type=text/html&node-id=022cd523-1d51-42ac-860a-7c210c9832a9&filename=assessment-march-3+%282%29.html&node-kind=file>

[nextjournal#output#6ccf655d-64b1-4c2d-86f0-6ac76bc04c48#result]:
<https://nextjournal.com/data/QmdM9Z1TSYU7iXnSb2gYwMtYRdzdqafcCqsVaTajFQZ2Jn?content-type=application/vnd.nextjournal.html%2Bhtml&node-id=6ccf655d-64b1-4c2d-86f0-6ac76bc04c48&node-kind=output>

[nextjournal#file#3bbf1eaa-c623-46ef-ba52-2b1bb8150636]:
<https://nextjournal.com/data/QmU1xW2DuHQSYPSE5euRgbLHX6G2x7f8B3Y2o3xNgGcFJ7?content-type=text/html&node-id=3bbf1eaa-c623-46ef-ba52-2b1bb8150636&filename=assessment-december-3.html&node-kind=file>

[nextjournal#file#6e9bae87-7207-42e9-957c-638e2e53cb5e]:
<https://nextjournal.com/data/QmZvHj1MsiDuHWDqariNnCPU7TQTmYpGocikuphpGV1rhp?content-type=text/html&node-id=6e9bae87-7207-42e9-957c-638e2e53cb5e&filename=assessment-december-31.html&node-kind=file>

[nextjournal#file#cceae52d-12b1-4fd2-9707-5f974bda6ffc]:
<https://nextjournal.com/data/QmVdfRwFZ8aTgFWeV3H19uC2uomFH3P5p8U5NxzBE8BNyd?content-type=text/html&node-id=cceae52d-12b1-4fd2-9707-5f974bda6ffc&filename=assessment-january-31-2023.html&node-kind=file>

[nextjournal#reference#242eafcf-3c9c-49e5-8fb5-f5faf0634d17]:
<#nextjournal#reference#242eafcf-3c9c-49e5-8fb5-f5faf0634d17>

[nextjournal#reference#1a0bf92b-b557-4cb0-ae9e-469664e73456]:
<#nextjournal#reference#1a0bf92b-b557-4cb0-ae9e-469664e73456>

[nextjournal#reference#b6a07561-ce98-440d-84bc-a3198d909ff0]:
<#nextjournal#reference#b6a07561-ce98-440d-84bc-a3198d909ff0>

[nextjournal#reference#dd1bff04-90f9-45e1-96d9-6e43f3743698]:
<#nextjournal#reference#dd1bff04-90f9-45e1-96d9-6e43f3743698>

[nextjournal#reference#525e9957-f106-4b93-a5cc-fa7ff73c811c]:
<#nextjournal#reference#525e9957-f106-4b93-a5cc-fa7ff73c811c>

[nextjournal#file#696ec1c6-2f92-4000-b7fe-069b4da30d8b]:
<https://nextjournal.com/data/QmbyqXxrEfHXi9gkL1aZxM5NLSo9v4mQKiiwCNiBEbN5Mt?content-type=application/x-zip-compressed&node-id=696ec1c6-2f92-4000-b7fe-069b4da30d8b&filename=out.zip&node-kind=file>

[nextjournal#reference#f0c20232-9d5e-4791-b966-2976173dc4aa]:
<#nextjournal#reference#f0c20232-9d5e-4791-b966-2976173dc4aa>

[nextjournal#reference#c437612b-f6d7-4f2f-b745-54a7d4487a9c]:
<#nextjournal#reference#c437612b-f6d7-4f2f-b745-54a7d4487a9c>

[nextjournal#output#ddcbe78e-ee11-4235-a250-c90d9dfe32d2#result]:
<https://nextjournal.com/data/QmU8cn33aeoNaPoqyZfVmPKZPga3JZdCeVrCUGyZKc129n?content-type=application/vnd.nextjournal.html%2Bhtml&node-id=ddcbe78e-ee11-4235-a250-c90d9dfe32d2&node-kind=output>

[nextjournal#output#ddcbe78e-ee11-4235-a250-c90d9dfe32d2#tfidf.pkl]:
<https://nextjournal.com/data/QmSmdA58vNTj4SUkVuB2Gp8cYTs1dXKAQVzVRo2dwfHmFW?content-type=application/octet-stream&node-id=ddcbe78e-ee11-4235-a250-c90d9dfe32d2&filename=tfidf.pkl&node-kind=output>

[nextjournal#output#f54371ae-8a6d-40d6-9e3a-f0469cb726b5#out_no_zero.zip]:
<https://nextjournal.com/data/QmagBo6woGjHs31fCBWQuBLJmD9eWVQNnWodpjuYcDhHo4?content-type=application/zip&node-id=f54371ae-8a6d-40d6-9e3a-f0469cb726b5&filename=out_no_zero.zip&node-kind=output>

[nextjournal#file#12ab85bc-e83e-4444-99d6-6750590b3557]:
<https://nextjournal.com/data/QmdcU8BwvncbTEscvZdQK5gJiJC4ncMTg97SZFt5CxJKfH?content-type=application/x-zip-compressed&node-id=12ab85bc-e83e-4444-99d6-6750590b3557&filename=test.zip&node-kind=file>

<details id="com.nextjournal.article">
<summary>This notebook was exported from <a href="https://nextjournal.com/a/RLpvVWAtLx7VuxE6Hg3dt?change-id=DMeHKGsexskN6urpQXwAdy">https://nextjournal.com/a/RLpvVWAtLx7VuxE6Hg3dt?change-id=DMeHKGsexskN6urpQXwAdy</a></summary>

```edn nextjournal-metadata
{:article
 {:settings {:numbered? false},
  :nodes
  {"022cd523-1d51-42ac-860a-7c210c9832a9"
   {:id "022cd523-1d51-42ac-860a-7c210c9832a9", :kind "file"},
   "12ab85bc-e83e-4444-99d6-6750590b3557"
   {:id "12ab85bc-e83e-4444-99d6-6750590b3557", :kind "file"},
   "1a0bf92b-b557-4cb0-ae9e-469664e73456"
   {:id "1a0bf92b-b557-4cb0-ae9e-469664e73456",
    :kind "reference",
    :link [:output "3bbf1eaa-c623-46ef-ba52-2b1bb8150636" nil]},
   "23d49574-c621-485a-9e75-1eb38413222a"
   {:compute-ref #uuid "9c12b0e1-8dd4-4b0a-b488-02761367227f",
    :exec-duration 2956,
    :id "23d49574-c621-485a-9e75-1eb38413222a",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "bab9f562-df84-4d17-9df0-d2b0a87cb385"]},
   "242eafcf-3c9c-49e5-8fb5-f5faf0634d17"
   {:id "242eafcf-3c9c-49e5-8fb5-f5faf0634d17",
    :kind "reference",
    :link [:output "022cd523-1d51-42ac-860a-7c210c9832a9" nil]},
   "3bbf1eaa-c623-46ef-ba52-2b1bb8150636"
   {:id "3bbf1eaa-c623-46ef-ba52-2b1bb8150636", :kind "file"},
   "525e9957-f106-4b93-a5cc-fa7ff73c811c"
   {:id "525e9957-f106-4b93-a5cc-fa7ff73c811c",
    :kind "reference",
    :link [:output "23d49574-c621-485a-9e75-1eb38413222a" "out.csv"]},
   "5ce2b639-b88c-4f3b-b087-928894f0effe"
   {:compute-ref #uuid "537d666c-3615-4d3a-a84f-04a359358c34",
    :exec-duration 12457,
    :id "5ce2b639-b88c-4f3b-b087-928894f0effe",
    :kind "code",
    :output-log-lines {:stdout 35},
    :runtime [:runtime "bab9f562-df84-4d17-9df0-d2b0a87cb385"]},
   "696ec1c6-2f92-4000-b7fe-069b4da30d8b"
   {:id "696ec1c6-2f92-4000-b7fe-069b4da30d8b", :kind "file"},
   "6a29be21-53aa-4577-98cc-772b5e7d8e1e"
   {:compute-ref #uuid "9c12b0e1-8dd4-4b0a-b488-02761367227f",
    :exec-duration 2956,
    :id "6a29be21-53aa-4577-98cc-772b5e7d8e1e",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "bab9f562-df84-4d17-9df0-d2b0a87cb385"]},
   "6ccf655d-64b1-4c2d-86f0-6ac76bc04c48"
   {:compute-ref #uuid "cdba95e4-cb13-4fa9-a6d2-b48ce69b9c4f",
    :exec-duration 327,
    :id "6ccf655d-64b1-4c2d-86f0-6ac76bc04c48",
    :kind "code",
    :output-log-lines {:stdout 3},
    :runtime [:runtime "bab9f562-df84-4d17-9df0-d2b0a87cb385"]},
   "6e9bae87-7207-42e9-957c-638e2e53cb5e"
   {:id "6e9bae87-7207-42e9-957c-638e2e53cb5e", :kind "file"},
   "a21f50e2-b9c6-43d7-8101-c0a47d6613ba"
   {:compute-ref #uuid "b5be1603-697e-48ee-b960-dda6a2c93955",
    :exec-duration 378,
    :id "a21f50e2-b9c6-43d7-8101-c0a47d6613ba",
    :kind "code",
    :output-log-lines {:stdout 9},
    :runtime [:runtime "bab9f562-df84-4d17-9df0-d2b0a87cb385"]},
   "b6a07561-ce98-440d-84bc-a3198d909ff0"
   {:id "b6a07561-ce98-440d-84bc-a3198d909ff0",
    :kind "reference",
    :link [:output "6e9bae87-7207-42e9-957c-638e2e53cb5e" nil]},
   "bab9f562-df84-4d17-9df0-d2b0a87cb385"
   {:environment
    [:environment
     {:article/nextjournal.id
      #uuid "5b45e08b-5b96-413e-84ed-f03b5b65bd66",
      :change/nextjournal.id
      #uuid "6034be04-2bc2-4468-af0e-cd7b9b6e7ed8",
      :node/id "0149f12a-08de-4f3d-9fd3-4b7a665e8624"}],
    :id "bab9f562-df84-4d17-9df0-d2b0a87cb385",
    :kind "runtime",
    :language "python",
    :resources {:machine-type "n1-standard-2"},
    :type :jupyter},
   "c437612b-f6d7-4f2f-b745-54a7d4487a9c"
   {:id "c437612b-f6d7-4f2f-b745-54a7d4487a9c",
    :kind "reference",
    :link [:output "12ab85bc-e83e-4444-99d6-6750590b3557" nil]},
   "c5e844d2-bd49-44e3-8a11-c0248e58549f"
   {:compute-ref #uuid "846f00d5-c56e-4dde-8ed3-56b105f284dc",
    :exec-duration 65,
    :id "c5e844d2-bd49-44e3-8a11-c0248e58549f",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "bab9f562-df84-4d17-9df0-d2b0a87cb385"],
    :stdout-collapsed? false},
   "cceae52d-12b1-4fd2-9707-5f974bda6ffc"
   {:id "cceae52d-12b1-4fd2-9707-5f974bda6ffc", :kind "file"},
   "dd1bff04-90f9-45e1-96d9-6e43f3743698"
   {:id "dd1bff04-90f9-45e1-96d9-6e43f3743698",
    :kind "reference",
    :link [:output "cceae52d-12b1-4fd2-9707-5f974bda6ffc" nil]},
   "ddcbe78e-ee11-4235-a250-c90d9dfe32d2"
   {:compute-ref #uuid "fd866d3e-3aa9-4401-b270-03f278ff73e4",
    :exec-duration 60495,
    :id "ddcbe78e-ee11-4235-a250-c90d9dfe32d2",
    :kind "code",
    :output-log-lines {:stdout 5},
    :runtime [:runtime "bab9f562-df84-4d17-9df0-d2b0a87cb385"]},
   "f0c20232-9d5e-4791-b966-2976173dc4aa"
   {:id "f0c20232-9d5e-4791-b966-2976173dc4aa",
    :kind "reference",
    :link [:output "696ec1c6-2f92-4000-b7fe-069b4da30d8b" nil]},
   "f54371ae-8a6d-40d6-9e3a-f0469cb726b5"
   {:compute-ref #uuid "fa3be652-c4d1-4b8b-85d8-4954867d68d2",
    :exec-duration 868,
    :id "f54371ae-8a6d-40d6-9e3a-f0469cb726b5",
    :kind "code",
    :output-log-lines {},
    :runtime [:runtime "bab9f562-df84-4d17-9df0-d2b0a87cb385"]}},
  :nextjournal/id #uuid "0366119f-6153-4cf1-9344-954a17acfa9b",
  :article/change
  {:nextjournal/id #uuid "640f83c5-9743-4b4e-b78e-6d716b5e0694"}}}

```
</details>
