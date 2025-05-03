import csv
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


with open('processed_data.csv') as csv_file:
    csv_read=csv.reader(csv_file, delimiter=',')
    
    X_text = list()
    Y_label = list()
    for lines in csv_read:
        X_text.append(lines[0])
        Y_label.append(int(lines[1]))

print(len(X_text))

X_train, X_test, Y_train, Y_test = train_test_split(X_text, Y_label, test_size=0.1)

vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, Y_train)

Y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

print("Accuracy: " + str(accuracy))

plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=[0,1], yticklabels=[0,1])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


