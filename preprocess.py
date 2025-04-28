import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp
import numpy as np
import pickle

# 1) load and drop the extra index
df = pd.read_csv("data/labeled_data.csv", index_col=0)

# 2) Binarize: 1 = toxic, 0 = non-toxic 
df["toxic"] = df["class"].apply(lambda c: 1 if c in [0,1] else 0)

# 3) Download stopwords once
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# 4) Define a cleaner
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)    # to remove the URLs
    text = re.sub(r"[^a-z\s]", "", text)          # keep letters + spaces
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# 5) Apply cleaning to the tweet column
df["clean_text"] = df["tweet"].astype(str).apply(clean)

# 6) Select only what we need and write out
out = df[["clean_text", "toxic"]]
out.to_csv("data/processed_data.csv", index=False, header=["text","label"])

print(f"→ Wrote data/processed_data.csv ({len(out)} rows)")

# 7) Tokenize & numeric features

# a) initialize 
vectorizer = CountVectorizer()

# b) transform
X = vectorizer.fit_transform(out["clean_text"])  # shape = (n_samples, n_features)
y = out["toxic"].values                         # shape = (n_samples,)

sp.save_npz("data/X_counts.npz", X)              # sparse feature matrix
np.save("data/y_labels.npy", y)            
with open("data/vectorizer.pkl", "wb") as f:     # save
    pickle.dump(vectorizer, f)

print(f"→ Saved data/X_counts.npz ({X.shape}), data/y_labels.npy, and data/vectorizer.pkl")
