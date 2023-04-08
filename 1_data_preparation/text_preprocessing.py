import nltk
import string
import re
from num2words import num2words

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_punctuation = string.punctuation
stop_words = set(nltk.corpus.stopwords.words("english"))


def to_lower_case(text):
    return "".join([i.lower() for i in text])


def remove_punctuation(text):
    return "".join([i for i in text if i not in stop_punctuation])


def remove_long_dash(text):
    return re.sub(r"—", " ", text)


def remove_urls(text):
    return re.sub(r"http\S+", "", text)


def remove_one_letter_words(tokens):
    return list(filter(lambda token: len(token) > 1, tokens))


def tokenize_text(text):
    return nltk.tokenize.word_tokenize(text)


def remove_stop_words(tokens):
    # avoid_stop_words = {"not", "n't", "no"}
    # stop_words = stop_words - avoid_stop_words
    return [i for i in tokens if i not in stop_words]


def do_stemming(tokens):
    ps = nltk.PorterStemmer()
    return [ps.stem(word) for word in tokens]


def do_lemmatization(tokens):
    wn = nltk.WordNetLemmatizer()
    return [wn.lemmatize(word) for word in tokens]


def remove_numeric_words(text):
    return re.sub(r"\S*\d+\S*", "", text)


def convert_nums_to_words(data):
    tokens = data
    new_text = []
    for word in tokens:
        if word.isdigit():
            if int(word) < 1000000000:
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
