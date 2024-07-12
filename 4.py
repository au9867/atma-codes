#4 Web Scraping
import requests
from bs4 import BeautifulSoup

def web_scraper(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            # Extract specific information from the webpage
            title = soup.title.string.strip()
            print("Title:", title)
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                print(p.get_text())
        else:
            print(f"Failed to retrieve: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")

if __name__ == "__main__":
    target_url = "https://gmail.com"  # Replace with the URL you want to scrape
    web_scraper(target_url)