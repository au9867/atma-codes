#11 Identifying the Topic of Scraped Data
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Step 1: Scrape data from a web page
url = "https://www.nytimes.com/"  # Replace with the URL of the webpage you want to scrape
response = requests.get(url)

# Step 2: Parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")
text_data = [p.get_text() for p in soup.find_all("p")]  # Extract text from paragraphs

# Step 3: Preprocess the text data (you can add more preprocessing steps)
cleaned_data = [text.lower() for text in text_data if text.strip()]  # Filter out empty strings

# Step 4: Transform text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.85, max_features=1000)
tfidf_matrix = vectorizer.fit_transform(cleaned_data)

# Step 5: Apply Latent Dirichlet Allocation (LDA) for topic modeling
num_topics = 5  # You can adjust the number of topics as needed
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(tfidf_matrix)

# Step 6: Print the topics and associated words
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx + 1}:")
    top_words_idx = topic.argsort()[-10:][::-1]  # Print top 10 words for each topic
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    print(" ".join(top_words))
