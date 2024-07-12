pip install textblob matplotlib seaborn wordcloud

#13 Twitter Data Analysis:
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd

# Example text data
texts = [
    "The product is amazing and exceeded my expectations!",
    "I had a terrible experience with the service.",
    "The new update is decent but could be better.",
    "I absolutely love the new features!",
    "This was a disappointing purchase, not worth the money.",
    "Great value for the price, highly recommend!",
    "The app crashes frequently, very frustrating.",
    "Customer support was very helpful and resolved my issue quickly."
]

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    return polarity, subjectivity

# Analyze sentiment of text data
sentiment_results = [analyze_sentiment(text) for text in texts]

# Count the number of positive, negative, and neutral texts
positive_count = sum(1 for p, _ in sentiment_results if p > 0)
negative_count = sum(1 for p, _ in sentiment_results if p < 0)
neutral_count = sum(1 for p, _ in sentiment_results if p == 0)

# Print sentiment counts
print(f"Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}")

# Plot sentiment distribution
labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive_count, negative_count, neutral_count]
colors = sns.color_palette("Set2", 3)  # Use a different color palette for the pie chart

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 14})
plt.title('Sentiment Distribution', fontsize=16)
plt.show()

# Create a bar chart for sentiment polarity
polarities = [polarity for polarity, _ in sentiment_results]
indices = np.arange(len(polarities))

plt.figure(figsize=(12, 6))
plt.bar(indices, polarities, color=['green' if p > 0 else 'red' if p < 0 else 'gray' for p in polarities])
plt.xticks(indices, texts, rotation=90, fontsize=10)
plt.title('Sentiment Polarity\n', fontsize=16)
plt.xlabel('Texts', fontsize=14)
plt.ylabel('Polarity', fontsize=14)
plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()

# Create a word cloud of common keywords
all_words = ' '.join(texts)
word_cloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.title('\nKeyword Cloud\n', fontsize=16)
plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()

# Extract most common keywords
words = all_words.split()
common_words = Counter(words).most_common(10)
print("\nMost common keywords:\n", common_words)

# Prepare data for keyword frequency plot
data = pd.DataFrame(common_words, columns=['Keyword', 'Frequency'])

# Plot keyword frequencies
plt.figure(figsize=(10, 6))
sns.barplot(x='Keyword', y='Frequency', hue='Keyword', data=data, palette='viridis', dodge=False, legend=False)
plt.title('\nTop 10 Most Common Keywords \n', fontsize=16)
plt.xlabel('Keywords', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()  # Adjust subplot parameters to give specified padding
plt.show()
