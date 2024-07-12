# Install necessary packages
!pip install nltk gensim

# Download NLTK data
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Import required libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models

# 2 Topic Detection for a Given Set of Corpus
corpus = [
    "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
    "Machine learning is a subset of artificial intelligence in the field of computer science that often uses statistical techniques to give computers the ability to learn with data.",
    "Natural Language Processing (NLP) is a subfield of artificial intelligence concerned with the interaction between computers and humans in natural language.",
    "Big data refers to extremely large datasets that are too large to be analyzed using traditional data processing techniques.",
    "Python is a popular programming language for data science and machine learning due to its simplicity and extensive libraries.",
    "Topic detection and clustering are essential tasks in text mining to discover the main themes and group similar documents together.",
]

# Preprocess the corpus
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return words

processed_corpus = [preprocess_text(text) for text in corpus]

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(processed_corpus)

# Create a document-term matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in processed_corpus]

# Build LDA model
num_topics = 2  # You can change the number of topics as per your requirement
lda_model = models.LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)

# Print the topics and their dominant words
print("Topics and their dominant words:")
for topic_id in range(num_topics):
    topic_words = lda_model.show_topic(topic_id)
    dominant_words = ", ".join([word for word, prob in topic_words])
    print(f"Topic {topic_id + 1}: {dominant_words}")
