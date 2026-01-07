import pandas as pd, numpy as np, json, re, string, random, nltk, spacy, plotly.graph_objects as go
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#STUDY CASE
from nltk.corpus import movie_reviews

# Build a list of (text, label) pairs
docs = [(movie_reviews.raw(fid), movie_reviews.categories(fid)[0])
        for fid in movie_reviews.fileids()]

# Shuffle for randomness but keep it reproducible
random.seed(42)
random.shuffle(docs)

texts, labels = zip(*docs)  # unzip into two lists

data_preview = pd.DataFrame({
    'Text': texts[:5],
    'Label': labels[:5]
})
pd.set_option('display.max_colwidth', 100)
print("Sample data:")
display(data_preview)


label_counts = pd.Series(labels).value_counts()

plt.figure(figsize=(6,4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='Set2')
plt.title("Class Distribution in Movie Reviews")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

from sklearn.model_selection import train_test_split

# 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)
print(f"Train size: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

baseline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", max_df=0.95)),
    ("clf", LogisticRegression(max_iter=200))
])

baseline.fit(X_train, y_train)
val_pred = baseline.predict(X_val)


from sklearn.metrics import ConfusionMatrixDisplay

print("Validation Set Performance:\n")
print(classification_report(y_val, val_pred))

cm = confusion_matrix(y_val, val_pred, labels=["pos", "neg"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["pos", "neg"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Baseline Model)")
plt.show()

import numpy as np

tfidf = baseline.named_steps["tfidf"]
clf   = baseline.named_steps["clf"]

feature_names = np.array(tfidf.get_feature_names_out())
coef = clf.coef_[0]

top_pos = feature_names[np.argsort(coef)][-15:][::-1]
top_neg = feature_names[np.argsort(coef)][:15][::-1]

print(f"Top features for POSITIVE reviews:\n{', '.join(top_pos)}\n")
print(f"Top features for NEGATIVE reviews:\n{', '.join(top_neg)}")

from sklearn.metrics import roc_auc_score

improved = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),     # include bigrams
        min_df=2,              # ignore rare words
        max_features=30000     # cap vocabulary size
    )),
    ("clf", LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        C=2.0
    ))
])

improved.fit(X_train, y_train)
val_pred = improved.predict(X_val)
val_proba = improved.predict_proba(X_val)[:,1]

print(classification_report(y_val, val_pred))
print("Validation ROC-AUC:", roc_auc_score([1 if y=='pos' else 0 for y in y_val], val_proba))

mis_idx = [i for i, (y, p) in enumerate(zip(y_val, val_pred)) if y != p]
print(f"Total misclassified samples: {len(mis_idx)}\n")

for i in mis_idx[:5]:
    print(f"TRUE={y_val[i]} | PRED={val_pred[i]}")
    print(X_val[i][:400], "\n---\n")

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    [1 if y=='pos' else 0 for y in y_val],
    val_proba,
    name="Improved Logistic Regression",
    color="darkorange"
)
plt.title("ROC Curve on Validation Set")
plt.show()

X_trainval = list(X_train) + list(X_val)
y_trainval = list(y_train) + list(y_val)

final_model = improved
final_model.fit(X_trainval, y_trainval)

test_pred = final_model.predict(X_test)

print("Final Test Performance:\n")
print(classification_report(y_test, test_pred))
cm_final = confusion_matrix(y_test, test_pred, labels=["pos", "neg"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=["pos", "neg"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix (Final Test)")
plt.show()