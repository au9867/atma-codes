#10 Scraping and Extracting Conversational Topics on Internet Forums
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Step 1: Define a target blog post URL (replace with another blog URL if desired)
blog_url = 'https://www.theguardian.com/us-news/article/2024/jul/04/biden-interview-debate-trump'

# Step 2: Fetch the HTML content of the blog post
response = requests.get(blog_url)

# Step 3: Extract blog post content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')
paragraphs = [p.text for p in soup.find_all('p')]

# Step 4: Preprocess text
def preprocess_text(text):
    return ' '.join(text.split())

paragraphs = [preprocess_text(p) for p in paragraphs if p.strip()]

# Step 5: Apply NLP techniques (TF-IDF and NMF) to identify topics
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(paragraphs)

nmf = NMF(n_components=5, random_state=1)
nmf.fit(tfidf)

# Step 6: Extract and print topics
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    topic_keywords = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(topic_keywords)}")
