# The essence of your requirement is: 👉 In the original agnews.csv (label, text) do Random Token Replacement (RTR) + Character-level disruption （TYPO） data augmentation and generation:
# label, text, text1, text2     Among: text：original sentence;   text1 / text2：Two random replacements enhance the results
# ✅ I. Core Idea (RTR) Standard Practice of RTR:
# Randomly select a portion of tokens from the sentence (for example, 10% to 20%)
# Replace these tokens with random words
# Instead of using [MASK], simply replace it with the words in vocab
# ✅ Second, directly provide you with runnable code (complete script)
# Second, directly provide you with runnable code (complete script): rtr_typo_augment.py
# ✅ III. Effect Description (Consistent with Your Requirements)
# input：
# 3, microsoft offer windows flavor retail market ...
# output：
# label,text,text1,text2
# 3,Original sentence, Enhanced sentence 1, Enhanced sentence 2
# Enhancement sentence example (RTR effect) :
# The words are randomly replaced
# The sentence structure remains
# Similar to the spelling disorder effect you gave (or even more "random")
# ✅Four. If you want to be closer to the "spelling error style" you gave
#
# Your current example is more like:
# 👉 Character-level perturbation（typo），It's not a pure RTR

# If you want it to be more like the effect you gave (for example, project → rpoject)

import pandas as pd
import random
import csv

# =========================
# parameter configuration
# =========================
REPLACE_PROB = 0.15   # Replacement ratio
AUG_NUM = 2           # Generate several enhanced sentences for each item

# =========================
# Build a vocabulary list (extract from data)
# =========================
def build_vocab(texts):
    vocab = set()
    for text in texts:
        words = text.split()
        vocab.update(words)
    return list(vocab)

# =========================
# RTR contrast enhancing function
# =========================
# I can add one for you:
#
# 👉 Character-level perturbation version (optional enhancement)
def typo_augment(word):
    if len(word) <= 3:
        return word

    i = random.randint(0, len(word)-2)
    word = list(word)
    word[i], word[i+1] = word[i+1], word[i]
    return "".join(word)

def random_token_replacement(text, vocab, replace_prob=0.15):
    words = text.split()
    new_words = []

    for word in words:
        # if random.random() < replace_prob:
        #     # Replace with random words
        #     new_word = random.choice(vocab)
        #     new_words.append(new_word)
        # else:
        #     new_words.append(word)
        if random.random() < replace_prob:
            if random.random() < 0.5:
                new_word = random.choice(vocab)  # RTR
                new_words.append(new_word)
            else:
                new_word = typo_augment(word)  # Replace with random words
                new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)

# =========================
# main function
# =========================
def augment_agnews(input_file, output_file):
    # reading data
    df = pd.read_csv(input_file, header=None, names=["label", "text"], encoding="ISO-8859-1")
    df = df[1:]
    texts = df["text"].fillna("NaN").astype(str).tolist()

    # Construct a word list
    vocab = build_vocab(texts)
    print(f"Vocab size: {len(vocab)}")

    results = []

    for idx, row in df.iterrows():
        label = row["label"]
        text = row["text"]

        # Generate two enhancement sentences
        aug_texts = []
        for _ in range(AUG_NUM):
            aug = random_token_replacement(text, vocab, REPLACE_PROB)
            aug_texts.append(aug)

        results.append([label, text, aug_texts[0], aug_texts[1]])

    # save file
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text", "text1", "text2"])
        writer.writerows(results)

    print(f"Saved to {output_file}")

def augment_stackoverflow(input_file, output_file):
    # reading data
    df = pd.read_csv(input_file, header=None, names=["label", "text", "Column3"], encoding="ISO-8859-1")
    df = df[1:]
    texts = df["text"].fillna("NaN").astype(str).tolist()

    # Construct a word list
    vocab = build_vocab(texts)
    print(f"Vocab size: {len(vocab)}")

    results = []

    for idx, row in df.iterrows():
        label = row["label"]
        text = row["text"]

        # Generate two enhancement sentences
        aug_texts = []
        for _ in range(AUG_NUM):
            aug = random_token_replacement(text, vocab, REPLACE_PROB)
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
    augment_agnews("dataset/agnews.csv", "dataset/augmentation/agnews_RTR_TYPO.csv")
    augment_agnews("dataset/biomedical.csv", "dataset/augmentation/biomedical_RTR_TYPO.csv")
    augment_agnews("dataset/googlenews_S.csv", "dataset/augmentation/googlenews_S_RTR_TYPO.csv")
    augment_agnews("dataset/googlenews_T.csv", "dataset/augmentation/googlenews_T_RTR_TYPO.csv")
    augment_agnews("dataset/googlenews_TS.csv", "dataset/augmentation/googlenews_TS_RTR_TYPO.csv")
    augment_agnews("dataset/searchsnippets.csv", "dataset/augmentation/searchsnippets_RTR_TYPO.csv")
    augment_stackoverflow("dataset/stackoverflow.csv", "dataset/augmentation/stackoverflow_RTR_TYPO.csv")
    augment_agnews("dataset/tweet.csv", "dataset/augmentation/tweet_RTR_TYPO.csv")

### Run Script
# python rtr_typo_augment.py