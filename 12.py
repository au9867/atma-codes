#12 Classification of a Scrapped data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
data = [
("Positive text", "positive"),
("Negative sentiment", "negative"),
("Neutral statement", "neutral"),
# Add more data samples with corresponding labels
]
texts, labels = zip(*data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
new_data = ["New text to classify"]
X_new = vectorizer.transform(new_data)
predicted_label = classifier.predict(X_new)
print(f"Predicted Label for New Data: {predicted_label[0]}")
