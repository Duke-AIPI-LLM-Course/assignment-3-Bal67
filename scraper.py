import requests
from bs4 import BeautifulSoup

# URL of Britannica page
URL = "https://www.britannica.com/science/diabetes-mellitus"

# Fetch the webpage
response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
if response.status_code != 200:
    print("Failed to fetch page")
    exit()

# Parse HTML
soup = BeautifulSoup(response.text, "html.parser")

# Extract paragraphs
text_data = " ".join([p.text for p in soup.find_all("p")])

# Save extracted text
with open("data.txt", "w", encoding="utf-8") as f:
    f.write(text_data)

print("✅ Web scraping completed. Data saved to data.txt")
