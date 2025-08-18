import json
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load links from JSON (you can also hardcode your list if needed)
links = [
    "https://raitamitra.karnataka.gov.in",
    "https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2002012",
    "https://dhtmanipur.mn.gov.in/schemes-project-implemented/",
    "https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=1658313#:~:text=For%20cotton%20season%202020%2D21,5550%2F%2D%20to%20Rs.",
    "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2146756",
    "https://fasal.haryana.gov.in/",
    "https://www.wtin.com/article/2025/february/03-02-25/india-s-budget-targets-cotton/#:~:text=The%20Budget%20announced%20an%20outlay,4417.03%20crore).",
    "https://services.india.gov.in/service/detail/major-schemes-for-farmers-1"
]

def fetch_text(url):
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # Remove scripts, styles, etc.
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return ""

# Step 1: Fetch all text
all_texts = []
for url in links:
    print(f"🌐 Fetching {url} ...")
    text = fetch_text(url)
    if text:
        all_texts.append({"url": url, "content": text})

# Step 2: Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = []
for entry in all_texts:
    docs = splitter.split_text(entry["content"])
    for i, chunk in enumerate(docs):
        chunks.append({
            "url": entry["url"],
            "chunk_id": i,
            "content": chunk
        })

# Step 3: Save chunks to JSON
with open("schemes_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(chunks)} chunks and saved to schemes_chunks.json")
