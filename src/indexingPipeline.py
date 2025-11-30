from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
import re

INDEX_NAME = "smart_docs"
es = Elasticsearch("http://localhost:9200")

mapping = {
    "settings": {
        "index": {
            "max_ngram_diff": 5
        },
        "analysis": {
            "char_filter": {
                "html_strip": {"type": "html_strip"}
            },
            "filter": {
                "length_filter": {"type": "length", "min": 3}
            },
            "tokenizer": {
                "autocomplete_infix_tokenizer": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 5,
                    "token_chars": ["letter", "digit"]
                }
            },
            "analyzer": {
                "autocomplete_infix": {
                    "type": "custom",
                    "tokenizer": "autocomplete_infix_tokenizer",
                    "filter": ["lowercase"]
                },
                "autocomplete_infix_search": {
                    "type": "custom",
                    "tokenizer": "lowercase"
                },
                "content_analyzer": {
                    "type": "custom",
                    "char_filter": ["html_strip"],
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "length_filter", "porter_stem"]
                }
            }
        }
    },

    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "autocomplete_infix",
                "search_analyzer": "autocomplete_infix_search"
            },
            "title_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "content": {"type": "text", "analyzer": "content_analyzer"},
            "author_raw": {"type": "keyword"},
            "dateline_raw": {"type": "text"},
            "date_raw": {"type": "text"},  # You are not converting this to real date
            "places": {"type": "keyword"},
            "temporalExpressions": {"type": "keyword"},
            "georeferences": {"type": "keyword"},

            "geopoints": {
                "type": "nested",
                "properties": {
                    "place": {"type": "keyword"},
                    "location": {"type": "geo_point"}
                }
            }
        }
    }
}

if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
es.indices.create(index=INDEX_NAME, body=mapping)

model = SentenceTransformer('all-MiniLM-L6-v2')

with open(r"smart-news-retrieval-\output\all_reuters_parsed.json", "r") as f:
    documents = json.load(f)

actions = []

for doc in documents:
    title = doc.get("title", "")
    content = doc.get("content", "")
    author_raw = doc.get("author_raw", "")
    dateline_raw = doc.get("dateline_raw", "")
    date_raw = doc.get("date_raw", "")

    places = doc.get("places", [])
    temporal = doc.get("temporalExpressions", [])
    georefs = doc.get("georeferences", [])

    # Convert geo points
    geopoints = []
    for g in doc.get("geopoints", []):
        geopoints.append({
            "place": g.get("place"),
            "location": {
                "lat": g.get("lat"),
                "lon": g.get("lon")
            }
        })

    # Encode title
    title_vector = model.encode(title).tolist()

    es_doc = {
        "_index": INDEX_NAME,
        "_source": {
            "title": title,
            "title_vector": title_vector,
            "content": content,
            "author_raw": author_raw,
            "dateline_raw": dateline_raw,
            "date_raw": date_raw,
            "places": places,
            "temporalExpressions": temporal,
            "georeferences": georefs,
            "geopoints": geopoints
        }
    }
    actions.append(es_doc)

helpers.bulk(es, actions)
print(f"Indexed {len(actions)} documents successfully.")

def hybrid_autocomplete(query, top_k=10):

    lexical_query = {
        "multi_match": {
            "query": query,
            "fields": ["title^3", "content"],
            "type": "bool_prefix"
        }
    }

    query_vector = model.encode(query).tolist()

    semantic_query = {
        "knn": {
            "field": "title_vector",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 50
        }
    }

    response = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        lexical_query,
                        {"constant_score": {"filter": semantic_query}}
                    ]
                }
            }
        }
    )

    return [hit["_source"]["title"] for hit in response["hits"]["hits"]]

# Test
print(hybrid_autocomplete("index removal"))
