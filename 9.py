#9 Information Extraction from Text Data
import spacy
import requests
from bs4 import BeautifulSoup

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define a function to extract named entities from the text
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Define a function to fetch information from Wikipedia
def get_wikipedia_summary(entity):
    base_url = "https://en.wikipedia.org/wiki/"
    url = base_url + entity.replace(" ", "_")
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        summary = paragraphs[0].get_text() if paragraphs else "No information found."
    else:
        summary = "No information found."
    return summary

# Example text to extract named entities
text = "Barack Obama was born in Honolulu, Hawaii. He served as the 44th President of the United States."

# Extract named entities from the text
entities = extract_named_entities(text)

# Display the named entities and retrieve information from Wikipedia
for entity, label in entities:
    print(f"Entity: {entity}, Label: {label}")
    if label == "PERSON":
        summary = get_wikipedia_summary(entity)
        print(f"\nWikipedia Summary: {summary}")
        
