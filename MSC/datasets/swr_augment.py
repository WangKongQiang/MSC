# Your requirement is essentially to change RTR (Random Word Replacement) to SWR (Stop-Words Replacement). The core difference is:
# 👉 Only replace the stop words (stop words), and replace them with "another stop word"
# I'll directly fix the code for you and explain the key points.
# ✅ The complete modified code (SWR version)
# Just replace your RTR function and related parts directly 👇
import pandas as pd
import random
import csv
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
# =========================
# parameter configuration
# =========================
REPLACE_PROB = 0.5   # ⚠️ SWR can generally be raised (because it only acts on stop words).
AUG_NUM = 2

# =========================
# Load stop word
# =========================
STOP_WORDS = set(stopwords.words('english'))

# =========================
# SWR contrast enhancing function
# =========================
def stopword_replacement(text, stop_words, replace_prob=0.5):
    words = text.split()
    new_words = []

    for word in words:
        # Only replace "stop word"
        if word.lower() in stop_words and random.random() < replace_prob:
            new_word = random.choice(list(stop_words))
            new_words.append(new_word)
        else:
            new_words.append(word)

    return " ".join(new_words)

# =========================
# Main function (General)
# =========================
def augment_data(input_file, output_file, has_extra_col=False):
    if has_extra_col:
        df = pd.read_csv(input_file, header=None,
                         names=["label", "text", "Column3"],
                         encoding="ISO-8859-1")
    else:
        df = pd.read_csv(input_file, header=None,
                         names=["label", "text"],
                         encoding="ISO-8859-1")

    df = df[1:]
    df["text"] = df["text"].fillna("NaN").astype(str)

    results = []

    for _, row in df.iterrows():
        label = row["label"]
        text = row["text"]

        aug_texts = []
        for _ in range(AUG_NUM):
            aug = stopword_replacement(text, STOP_WORDS, REPLACE_PROB)
            aug_texts.append(aug)

        results.append([label, text, aug_texts[0], aug_texts[1]])

    # save file
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text", "text1", "text2"])
        writer.writerows(results)

    print(f"Saved to {output_file}")

# =========================
# execute
# =========================
if __name__ == "__main__":
    augment_data("dataset/agnews.csv", "dataset/augmentation/agnews_SWR.csv")
    augment_data("dataset/biomedical.csv", "dataset/augmentation/biomedical_SWR.csv")
    augment_data("dataset/googlenews_S.csv", "dataset/augmentation/googlenews_S_SWR.csv")
    augment_data("dataset/googlenews_T.csv", "dataset/augmentation/googlenews_T_SWR.csv")
    augment_data("dataset/googlenews_TS.csv", "dataset/augmentation/googlenews_TS_SWR.csv")
    augment_data("dataset/searchsnippets.csv", "dataset/augmentation/searchsnippets_SWR.csv")
    augment_data("dataset/stackoverflow.csv", "dataset/augmentation/stackoverflow_SWR.csv", has_extra_col=True)
    augment_data("dataset/tweet.csv", "dataset/augmentation/tweet_SWR.csv")

    ### Run Script
    # python swr_augment.py