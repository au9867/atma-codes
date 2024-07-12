pip install tweepy

#14 Classification of Twitter Sentiments
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns

# Example list of text data
texts = [
    "I love programming in Python!",
    "The weather today is terrible.",
    "I just bought a new laptop and it's amazing.",
    "I am feeling neutral about this project."
]

# Function to analyze sentiment
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Analyze sentiment of text data
sentiments = [analyze_sentiment(text) for text in texts]

# Count the number of positive, negative, and neutral sentiments
positive_texts = sentiments.count('Positive')
negative_texts = sentiments.count('Negative')
neutral_texts = sentiments.count('Neutral')

# Plot the sentiment distribution (Pie Chart)
plt.figure(figsize=(8, 6))
labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive_texts, negative_texts, neutral_texts]
pie_palette = sns.color_palette("viridis", 3)
plt.pie(sizes, labels=labels, colors=pie_palette, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title("\nSentiment Distribution (Pie Chart)\n")
plt.show()

# Plot the sentiment distribution (Bar Chart)
plt.figure(figsize=(8, 6))
bar_palette = sns.color_palette("pastel", 3)
sns.barplot(x=labels, y=sizes, palette=bar_palette, hue=labels, dodge=False, legend=False)
plt.title("\nSentiment Distribution (Bar Chart)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Create a word cloud
all_texts = ' '.join(texts)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(all_texts)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("\nWord Cloud of Text Data\n")
plt.show()

# Create a DataFrame for text polarity and subjectivity
polarity_subjectivity = [(TextBlob(text).sentiment.polarity, TextBlob(text).sentiment.subjectivity) for text in texts]
df = pd.DataFrame(polarity_subjectivity, columns=["Polarity", "Subjectivity"])

# Plot Polarity and Subjectivity
plt.figure(figsize=(10, 6))
scatter_palette = sns.color_palette("coolwarm", 3)
sns.scatterplot(x="Polarity", y="Subjectivity", data=df, hue=sentiments, palette=scatter_palette, s=100)
plt.title("\nPolarity and Subjectivity of Text Data")
plt.xlabel("\nPolarity")
plt.ylabel("Subjectivity")
plt.legend(title="Sentiment")
plt.show()

# Create a boxplot for polarity
plt.figure(figsize=(8, 6))
box_palette = sns.color_palette("Set2", 3)
sns.boxplot(x=sentiments, y=df["Polarity"], palette=box_palette, hue=sentiments, dodge=False)
plt.title("\nBoxplot of Polarity by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Polarity")
plt.legend(title="Sentiment", loc='upper right')
plt.show()

# Create a heatmap of sentiment counts
sentiment_counts = pd.DataFrame({'Sentiment': labels, 'Count': sizes})
plt.figure(figsize=(8, 6))
heatmap_palette = sns.light_palette("seagreen", as_cmap=True)
sns.heatmap(sentiment_counts.pivot_table(index='Sentiment', values='Count'), annot=True, cmap=heatmap_palette, fmt='d')
plt.title("\nHeatmap of Sentiment Counts")
plt.show()

# Plot the frequency of keywords (example)
keywords = ['python', 'weather', 'laptop', 'project']
frequency = [4, 3, 2, 2]
keyword_df = pd.DataFrame({'Keywords': keywords, 'Frequency': frequency})
plt.figure(figsize=(10, 6))
keyword_palette = sns.color_palette("muted", 4)
sns.barplot(x='Keywords', y='Frequency', data=keyword_df, palette=keyword_palette, hue='Keywords', dodge=False, legend=False)
plt.title("\nTop 10 Most Common Keywords")
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.show()

