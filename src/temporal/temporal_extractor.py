import re
import spacy
import dateparser

nlp = spacy.load("en_core_web_sm")

def extract_temporal_expressions(text):
    if not text:
        return []

    temporal_set = set()
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "DATE":
            temporal_set.add(ent.text.strip())

    regex_patterns = [
        r"\b\d{4}/\d{2}\b",
        r"\b\d{4}/\d{4}\b",
        r"\b\d{4}\b",
        r"\bQ[1-4]\b",
        r"\bfirst quarter\b",
        r"\bsecond quarter\b",
        r"\bthird quarter\b",
        r"\bfourth quarter\b",
        r"\bearly \w+\b",
        r"\blate \w+\b",
        r"\bmid \w+\b",
        r"\b\d{1,2}-[A-Z]{3}-\d{4}\b"
    ]

    for pattern in regex_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for m in matches:
            temporal_set.add(m.strip())

    return list(temporal_set)
