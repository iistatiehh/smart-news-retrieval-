 
# %pip install elasticsearch sentence-transformers
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
import time


 
INDEX_NAME = "news_reuters_docs"
es = Elasticsearch(
    "http://localhost:9200",
    request_timeout=60,
    max_retries=3,
    retry_on_timeout=True
)
model = SentenceTransformer('all-MiniLM-L6-v2')


 

# Add this to clean up old indices
print("Cleaning up...")
try:
    # Delete old problematic indices
    all_indices = es.cat.indices(format='json')
    for idx in all_indices:
        if idx['health'] == 'red' or idx['status'] == 'close':
            print(f"Deleting problematic index: {idx['index']}")
            es.indices.delete(index=idx['index'], ignore=[400, 404])
except:
    pass

print(es.ping())

 
mapping ={
  "settings": {
    "index": {
      "max_ngram_diff": 5
    },
    "analysis": {
      "char_filter": {
        "html_strip": { "type": "html_strip" }
      },
      "filter": {
        "length_filter": { "type": "length", "min": 3 }
      },
      "tokenizer": {
        "autocomplete_infix_tokenizer": {
          "type": "ngram",
          "min_gram": 3,
          "max_gram": 8,
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
        "search_analyzer": "autocomplete_infix_search",
        "fields": {
          "keyword": { "type": "keyword" }
        }
      },
      "title_vector": {
        "type": "dense_vector",
        "dims": 384,
        "index": True,
        "similarity": "cosine",
        "index_options": { "type": "hnsw", "m": 16, "ef_construction": 100 }
      },
      "content_chunks": {
        "type": "nested",
        "properties": {
          "text": { "type": "text" },
          "vector": {
            "type": "dense_vector",
            "dims": 384,
            "index": True,
            "similarity": "cosine",
            "index_options": { "type": "hnsw", "m": 16, "ef_construction": 100 }
          }
        }
      },
      "content": {
        "type": "text",
        "analyzer": "content_analyzer",
        "fields": {
          "keyword": { "type": "keyword" }
        }
      },
      "authors": {
        "type": "nested",
        "properties": {
          "first_name": { "type": "text" },
          "last_name": { "type": "text" },
          "email": { "type": "keyword" }
        }
      },
      "date": {
        "type": "date",
        "format": "strict_date_optional_time||yyyy-MM-dd'T'HH:mm:ss||epoch_millis"
      },
      "dateline": { "type": "text" },
      "geo_location": { "type": "geo_point" },
      "temporalExpressions": { "type": "keyword" },
      "georeferences": { "type": "keyword" },
      "places": { "type": "keyword" },
      "geopoints": {
        "type": "nested",
        "properties": {
          "place": { "type": "keyword" },
          "location": { "type": "geo_point" }
        }
      },
      "topics": { "type": "keyword" },
      "people": { "type": "keyword" },
      "orgs": { "type": "keyword" },
      "exchanges": { "type": "keyword" },
      "companies": { "type": "keyword" }
    }
  }
}


 
print(es.ping())


 
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
es.indices.create(index=INDEX_NAME, body=mapping)

# Add this - wait for index to be ready
print("Waiting for index to be ready...")
time.sleep(5)
es.cluster.health(wait_for_status='yellow', timeout='30s')


 
import json, time, re


file_path = r"C:\Users\asus\Desktop\NewsRetrival\smart-news-retrieval-\output\reuters_full.json"

def chunk_text(text, max_len=500):
    """Split text into chunks of max_len characters."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) <= max_len:
            current_chunk += " " + sent if current_chunk else sent
        else:
            chunks.append(current_chunk)
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

print("Loading documents...")
with open(file_path, "r", encoding="utf-8") as f:
    documents = json.load(f)
print(f"Loaded {len(documents)} documents")

batch_size = 500
total_indexed = 0

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    actions = []

    print(f"Processing batch {i//batch_size + 1} ({i} to {min(i+batch_size, len(documents))})")

    for doc in batch:
        title = doc.get("title", "")
        content = doc.get("content", "")
        date = doc.get("date")
        dateline = doc.get("dateline", "")
        places = doc.get("places", [])
        temporal = doc.get("temporalExpressions", [])
        georefs = doc.get("georeferences", [])

        authors = []
        for author in doc.get("authors", []):
            if author:
                authors.append({
                    "first_name": author.get("first_name", ""),
                    "last_name": author.get("last_name", ""),
                    "email": author.get("email", "")
                })

        geopoints = []
        for g in doc.get("geopoints", []):
            if g and g.get("location"):
                loc = g["location"]
                lat = loc.get("lat")
                lon = loc.get("lon")
                place = g.get("place", "")
                
                if lat is not None and lon is not None:
                    geopoints.append({
                        "place": place,
                        "location": {"lat": lat, "lon": lon}
                    })
        
        geo_location = None
        if geopoints:
            geo_location = geopoints[0]["location"]
        
        # title vector
        title_vector = model.encode(title).tolist()

        # chunk content and encode each chunk
        content_chunks = chunk_text(content, max_len=500)
        content_chunks_encoded = [
            {"text": chunk, "vector": model.encode(chunk).tolist()} for chunk in content_chunks
        ]

        es_doc = {
            "_index": INDEX_NAME,
            "_source": {
                "title": title,
                "title_vector": title_vector,
                "content": content,
                "content_chunks": content_chunks_encoded,
                "authors": authors,
                "date": date,
                "dateline": dateline,
                "places": places,
                "temporalExpressions": temporal,
                "georeferences": georefs,
                "geopoints": geopoints,
                "geo_location": geo_location
            }
        }
        actions.append(es_doc)

    try:
        success, failed = helpers.bulk(es, actions, raise_on_error=False, request_timeout=120)
        total_indexed += success
        print(f"  Indexed {success} documents (Failed: {len(failed)})")
        if failed:
            print(f"  First error: {failed[0]}")
        time.sleep(1)
    except Exception as e:
        print(f"  Error in batch: {e}")
        continue

print(f"\nTotal indexed: {total_indexed} documents successfully.")
es.indices.refresh(index=INDEX_NAME)

 
def hybrid_search(query, user_location=None, temporal_filter=None, top_k=10):
    """
    Enhanced hybrid search with fuzzy matching, recency boosting, and location-aware ranking.
    
    Args:
        query: Search query string
        user_location: Dict with 'lat' and 'lon' for geo-proximity ranking (optional)
        temporal_filter: Dict with 'start' and 'end' dates for temporal filtering (optional)
        top_k: Number of results to return
    """
    query_vector = model.encode(query).tolist()
    
    # Build the base query with function_score for recency and location boosting
    query_body = {
        "size": top_k,
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^3", "content"],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO",  # Handles typos (1-2 char edits)
                                    "prefix_length": 2,   # First 2 chars must match exactly
                                    "max_expansions": 50  # Limit fuzzy term expansions
                                }
                            }
                        ]
                    }
                },
                "functions": [
                    # Recency boost - exponential decay favoring recent documents
                    {
                        "exp": {
                            "date": {
                                "origin": "now",
                                "scale": "90d",    # Documents 90 days old get 50% weight
                                "offset": "7d",    # No decay for documents within 7 days
                                "decay": 0.5
                            }
                        },
                        "weight": 2.0  # Increase importance of recency
                    }
                ],
                "score_mode": "sum",      # Add function scores together
                "boost_mode": "multiply"  # Multiply with query score
            }
        },
        "knn": [
            {
                "field": "title_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 100,
                "boost": 2.0  # Title vectors weighted higher
            },
            {
                "field": "content_chunks.vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 100,
                "boost": 1.0
            }
        ]
    }
    
    # Add geo-proximity boost if user location is provided
    if user_location and "lat" in user_location and "lon" in user_location:
        query_body["query"]["function_score"]["functions"].append({
            "gauss": {
                "geo_location": {
                    "origin": {
                        "lat": user_location["lat"],
                        "lon": user_location["lon"]
                    },
                    "scale": "200km",   # Documents 200km away get 50% weight
                    "offset": "50km",   # No decay within 50km
                    "decay": 0.5
                }
            },
            "weight": 1.5  # Adjust importance of location proximity
        })
    
    # Add temporal range filter if provided
    if temporal_filter and "start" in temporal_filter and "end" in temporal_filter:
        query_body["query"]["function_score"]["query"]["bool"]["filter"] = [
            {
                "range": {
                    "date": {
                        "gte": temporal_filter["start"],
                        "lte": temporal_filter["end"]
                    }
                }
            }
        ]
    
    # Execute search
    response = es.search(
        index=INDEX_NAME,
        body=query_body,
        request_timeout=30
    )
    
    # Return results with full metadata
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "title": hit["_source"]["title"],
            "score": hit["_score"],
            "date": hit["_source"].get("date"),
            "geo_location": hit["_source"].get("geo_location"),
            "places": hit["_source"].get("places", []),
            "authors": hit["_source"].get("authors", []),
            "content_preview": hit["_source"]["content"][:200] + "..."  # First 200 chars
        })
    
    return results


# Test cases demonstrating all features
print("\n=== Test 1: Basic search ===")
results = hybrid_search("banking crisis")
print(f"Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']} (score: {result['score']:.2f})")

print("\n=== Test 2: Fuzzy search (with typo) ===")
results = hybrid_search("bankng crsis")  # Intentional typos
print(f"Found {len(results)} results with typos corrected:")
for i, result in enumerate(results[:3], 1):
    print(f"{i}. {result['title']}")

print("\n=== Test 3: Location-aware search ===")
london_location = {"lat": 51.5074, "lon": -0.1278}
results = hybrid_search("banking crisis", user_location=london_location)
print(f"Found {len(results)} results near London:")
for i, result in enumerate(results[:3], 1):
    location = result.get('geo_location', 'N/A')
    print(f"{i}. {result['title']}")
    print(f"   Location: {location}")

print("\n=== Test 4: Temporal search (1987 documents) ===")
temporal_filter = {
    "start": "1987-01-01",
    "end": "1987-12-31"
}
results = hybrid_search("banking", temporal_filter=temporal_filter)
print(f"Found {len(results)} results from 1987:")
for i, result in enumerate(results[:3], 1):
    print(f"{i}. {result['title']}")
    print(f"   Date: {result.get('date', 'N/A')}")

print("\n=== Test 5: Combined spatio-temporal query ===")
# Search for German news near Hamburg in March 1987
hamburg_location = {"lat": 53.5511, "lon": 9.9937}
march_1987 = {
    "start": "1987-03-01",
    "end": "1987-03-31"
}
results = hybrid_search(
    query="Germany inflation", 
    user_location=hamburg_location,
    temporal_filter=march_1987,
    top_k=5
)
print(f"Found {len(results)} German news near Hamburg in March 1987:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['title']}")
    print(f"   Date: {result.get('date', 'N/A')}")
    print(f"   Places: {', '.join(result.get('places', []))}")
    print()


