import os
from bs4 import BeautifulSoup
import json
import re
from src.temporal.temporal_extractor import extract_temporal_expressions


DATA_DIR = "/Users/mac2/University/Information Retrieval/Smart Doc System/smart-news-retrieval-/archive"
OUTPUT_DIR = "/Users/mac2/University/Information Retrieval/Smart Doc System/smart-news-retrieval-/output"


# ============================
#   ADVANCED AUTHOR EXTRACTOR
# ============================

def extract_author_from_body(body_text):

    if not body_text:
        return None

    # split content into lines
    lines = body_text.strip().split("\n")

    # we search only the LAST 6 lines (most author lines appear at the end)
    search_lines = reversed(lines[-6:])

    for line in search_lines:
        clean = line.strip()

        # -----------------------------------------
        # 1) "Reporting by John Smith"
        # -----------------------------------------
        match = re.search(r"reporting by (.+)", clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # -----------------------------------------
        # 2) "Editing by Jane Doe"
        # -----------------------------------------
        match = re.search(r"editing by (.+)", clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # -----------------------------------------
        # 3) "Written by Michael Brown"
        # -----------------------------------------
        match = re.search(r"written by (.+)", clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # -----------------------------------------
        # 4) "By Mark Johnson"
        # -----------------------------------------
        match = re.search(r"^by (.+)", clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # -----------------------------------------
        # 5) "- John Smith, Reuters"
        # -----------------------------------------
        match = re.search(r"^-?\s*([A-Z][a-z]+(?: [A-Z][a-z]+)*,\s*reuters?)", clean, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # -----------------------------------------
        # 6) "John Smith, Reuters"
        #    OR "John Smith Reuters"
        # -----------------------------------------
        match = re.search(
            r"([A-Z][a-z]+ [A-Z][a-z]+.*reuters?)",
            clean,
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        # -----------------------------------------
        # 7) only "Reuter" or "Reuters"
        # -----------------------------------------
        if clean.lower() in ["reuter", "reuters"]:
            return clean

        # -----------------------------------------
        # 8) Lone proper name at the end
        #    e.g., "John Smith"
        # -----------------------------------------
        match = re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+", clean)
        if match:
            return match.group(0).strip()

    return None


# ============================
#        PARSER
# ============================

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

        # ---- content extraction ----
        body = None
        if text_tag:
            body_tag = text_tag.find("body")
            if body_tag:
                body = body_tag.get_text(" ", strip=True)
            else:
                body = text_tag.get_text(" ", strip=True)

        # ---- author extraction ----
        if author_tag:
            author_raw = author_tag.get_text(strip=True)
        else:
            author_raw = extract_author_from_body(body)

        # ---- other fields ----
        date_raw = date_tag.get_text(strip=True) if date_tag else None
        dateline_raw = dateline_tag.get_text(" ", strip=True) if dateline_tag else None

        places = []
        if places_tag:
            for d in places_tag.find_all("d"):
                place_name = d.get_text(strip=True)
                if place_name:
                    places.append(place_name)
            
        temporal_expressions = extract_temporal_expressions(body)


        doc = {
            "title": title,
            "content": body,
            "author_raw": author_raw,
            "date_raw": date_raw,
            "dateline_raw": dateline_raw,
            "places": places,
            "temporalExpressions": temporal_expressions
        }

        if title or body:
            docs.append(doc)

    return docs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_docs = []

    # Loop over all .sgm files in the archive directory
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".sgm"):
            file_path = os.path.join(DATA_DIR, filename)
            print(f"Processing file: {filename}")

            docs = parse_reuters_file(file_path)
            print(f" -> Extracted {len(docs)} documents")

            all_docs.extend(docs)

    # Save all documents into ONE json file
    output_path = os.path.join(OUTPUT_DIR, "all_reuters_parsed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, ensure_ascii=False, indent=2)

    print("\n========================================")
    print(f"Total documents extracted: {len(all_docs)}")
    print(f"Saved combined output to: {output_path}")
    print("========================================")

if __name__ == "__main__":
    main()
