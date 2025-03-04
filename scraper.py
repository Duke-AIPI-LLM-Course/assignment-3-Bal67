import requests
from bs4 import BeautifulSoup

URL = "https://www.who.int/news-room/fact-sheets/detail/diabetes"
response = requests.get(URL)
html_content = response.text

soup = BeautifulSoup(html_content, "html.parser")
text_data = " ".join([p.text for p in soup.find_all("p")])

with open("data.txt", "w", encoding="utf-8") as f:
    f.write(text_data)

print("Web scraping completed. Data saved to data.txt.")
