#6 Web Indexer
import requests
from bs4 import BeautifulSoup
import re

def index_web_page(url, index):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            words = re.findall(r'\b\w+\b', text.lower())  # Extract words from text
            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(url)
            print(f"Indexed: {url}")
        else:
            print(f"Failed to retrieve: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")

if __name__ == "__main__":
    starting_url = "https://google.com"  # Replace with your starting URL
    max_pages = 10  # Adjust as needed
    index = {}

    # Crawl and index web pages
    visited_urls = set()
    queue = [starting_url]

    while queue and len(visited_urls) < max_pages:
        url = queue.pop(0)
        if url not in visited_urls:
            visited_urls.add(url)
            index_web_page(url, index)
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    for link in soup.find_all("a", href=True):
                        new_url = link["href"]
                        if new_url.startswith("http"):
                            queue.append(new_url)
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")

    # Print indexed words and their associated URLs
    for word, urls in index.items():
        print(f"{word}: {urls}")

