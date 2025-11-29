import os
from bs4 import BeautifulSoup
import json
import re
from src.temporal.temporal_extractor import extract_temporal_expressions
from src.geo.georeference_extractor import extract_georeferences



DATA_DIR = "/Users/mac2/University/Information Retrieval/Smart Doc System/smart-news-retrieval-/archive"
OUTPUT_DIR = "/Users/mac2/University/Information Retrieval/Smart Doc System/smart-news-retrieval-/output"


def extract_author_from_body(body_text):

    if not body_text:
        return None

    lines = body_text.strip().split("\n")
    search_lines = reversed(lines[-6:])

    for line in search_lines:
        clean = line.strip()

        patterns = [
            (r"reporting by (.+)", 1),
            (r"editing by (.+)", 1),
            (r"written by (.+)", 1),
            (r"^by (.+)", 1),
            (r"^-?\s*([A-Z][a-z]+(?: [A-Z][a-z]+)*,\s*reuters?)", 1),
            (r"([A-Z][a-z]+ [A-Z][a-z]+.*reuters?)", 1),
            (r"[A-Z][a-z]+ [A-Z][a-z]+", 0)
        ]

        for pattern, group in patterns:
            match = re.search(pattern, clean, re.IGNORECASE)
            if match:
                return match.group(group).strip()

        if clean.lower() in ["reuter", "reuters"]:
            return clean

    return None


def parse_reuters_file(file_path):

    with open(file_path, "r", encoding="latin-1") as f:
        raw_text = f.read()

    soup = BeautifulSoup(raw_text, "html.parser")
    docs = []

    for reuter in soup.find_all("reuters"):
        title_tag = reuter.find("title")
        text_tag = reuter.find("text")
        author_tag = reuter.find("author")
        date_tag = reuter.find("date")
        places_tag = reuter.find("places")
        dateline_tag = reuter.find("dateline")

        title = title_tag.get_text(strip=True) if title_tag else None

        body = None
        if text_tag:
            body_tag = text_tag.find("body")
            if body_tag:
                body = body_tag.get_text(" ", strip=True)
            else:
                body = text_tag.get_text(" ", strip=True)

        if author_tag:
            author_raw = author_tag.get_text(strip=True)
        else:
            author_raw = extract_author_from_body(body)

        date_raw = date_tag.get_text(strip=True) if date_tag else None
        dateline_raw = dateline_tag.get_text(" ", strip=True) if dateline_tag else None

        places = []
        if places_tag:
            for d in places_tag.find_all("d"):
                place_name = d.get_text(strip=True)
                if place_name:
                    places.append(place_name)

        # 🔥 NEW — TEMPORAL + GEO
        temporal_expressions = extract_temporal_expressions(body or "")
        geo_references = extract_georeferences(body or "")

        # 🔥 Rule 1 — if no geo found from text, fallback to places
        if not geo_references and places:
            geo_references = places.copy()

        # 🔥 Rule 2 — if still empty, extract from dateline (e.g., "BOSTON, March 11 -")
        if not geo_references and dateline_raw:
            
            city = dateline_raw.split(",")[0].strip()
            if city and city.isalpha():
                geo_references = [city]

        # 🔥 Rule 3 — if still empty, assign ["Unknown"]
        if not geo_references:
            geo_references = ["Unknown"]


        doc = {
            "title": title,
            "content": body,
            "author_raw": author_raw,
            "date_raw": date_raw,
            "dateline_raw": dateline_raw,
            "places": places,
            "temporalExpressions": temporal_expressions,
            "georeferences": geo_references
        }

        if title or body:
            docs.append(doc)

    return docs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_docs = []

    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".sgm"):
            file_path = os.path.join(DATA_DIR, filename)
            print(f"Processing file: {filename}")

            docs = parse_reuters_file(file_path)
            print(f" -> Extracted {len(docs)} documents")

            all_docs.extend(docs)

    output_path = os.path.join(OUTPUT_DIR, "all_reuters_parsed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print("\n========================================")
    print(f"Total documents extracted: {len(all_docs)}")
    print(f"Saved combined output to: {output_path}")
    print("========================================")


if __name__ == "__main__":
    main()
