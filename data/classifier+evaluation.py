import csv
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# 1. Load CSV using pandas (simpler than csv.reader)
with open('processed_data.csv') as csv_file:
    csv_read=csv.reader(csv_file, delimiter=',')
    next(csv_read)
    # 2. Extract features and labels
    X_text = list()
    Y_label = list()
    for lines in csv_read:
        X_text.append(lines[0])
        Y_label.append(int(lines[1]))

print(len(X_text))
# 3. Split the data into training and testing sets (with fixed seed for reproducibility)
X_train, X_test, Y_train, Y_test = train_test_split(X_text, Y_label, test_size=0.1)
print(len(X_test))
# 4. Convert text into numeric vectors
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Train a Naive Bayes model
model = BernoulliNB()
model.fit(X_train_vectorized, Y_train)

# 6. Predict on test data
Y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

# 8. Print metrics
print("Evaluation Metrics:")
print(f"→ Accuracy : {accuracy:.4f}")
print(f"→ Precision: {precision:.4f}")
print(f"→ Recall   : {recall:.4f}")
print(f"→ F1 Score : {f1:.4f}")

# 9. Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=[0,1], yticklabels=[0,1])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()


