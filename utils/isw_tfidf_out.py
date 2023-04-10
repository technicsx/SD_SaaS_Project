import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def isw_tfidf_output(filtered_df, frequent_words, __builtins__, df, features_count):
    filtered_df = filtered_df.apply(
        lambda d: " ".join(w for w in d.split() if w not in frequent_words)
    )

    df["Tokens"] = filtered_df

    filenames = df["Name"]
    dates = df["Date"]

    tfidf = TfidfVectorizer(smooth_idf=True, use_idf=True, max_features=features_count)
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

    output_name = "results/tfidf_" + str(features_count) + ".csv"

    res_df.to_csv(output_name, index=False)
    print("Successfully written .to_csv")
