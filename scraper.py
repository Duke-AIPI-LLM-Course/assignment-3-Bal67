import requests
from bs4 import BeautifulSoup

# URL of Britannica page
URL = "https://www.britannica.com/science/diabetes-mellitus"


response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
if response.status_code != 200:
    print("Failed to fetch page")
    exit()


soup = BeautifulSoup(response.text, "html.parser")


text_data = " ".join([p.text for p in soup.find_all("p")])

with open("data.txt", "w", encoding="utf-8") as f:
    f.write(text_data)

print("Web scraping completed. Data saved to data.txt")
