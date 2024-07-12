#7 Information visualization of text data
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text_data = "In the rapidly changing world of technology, staying up to date with the latest trends and innovations is crucial for businesses. Companies must adapt to emerging technologies, such as artificial intelligence (AI), machine learning, and blockchain, to remain competitive. AI, for instance, is transforming various industries, from healthcare to finance. Machine learning is being used to analyze large datasets for actionable insights. Additionally, blockchain technology is revolutionizing supply chain management and ensuring transparency."


# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.show()
