#15 Classification of Facebook Data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Sample data (replace with your Facebook dataset) 
data = {
'text': ['Post 1 content', 'Comment on Post 1', 'Another post', 'Comment on Another post'],
'category': ['Post', 'Comment', 'Post', 'Comment'] }

# Create a DataFrame
df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['category'], test_size=0.2, random_state=42)

# Create and train a classifier (Naive Bayes used here as an example) 
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions) 
report = classification_report(y_test, predictions)

print(f"Accuracy: {accuracy}") 
print("Classification Report:\n", report)