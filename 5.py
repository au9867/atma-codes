#5 Web Crawling
import requests
from bs4 import BeautifulSoup

def web_crawler(url, max_depth=3):
    visited_urls = set()

    def crawl(url, depth):
        if depth > max_depth:
            return
        if url in visited_urls:
            return
        print(f"Crawling {url}")
        visited_urls.add(url)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                for link in soup.find_all("a", href=True):
                    new_url = link["href"]
                    if new_url.startswith("http"):
                        crawl(new_url, depth + 1)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")

    crawl(url, 0)

if __name__ == "__main__":
    starting_url = "https://google.com"  # Replace with the starting URL
    web_crawler(starting_url)
