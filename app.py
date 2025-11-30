from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

ES_HOST = "http://localhost:9200"
INDEX_NAME = "smart_docs"
es = Elasticsearch(ES_HOST)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

app = FastAPI(title="Semantic + Lexical Autocomplete API")

# add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/autocomplete")
def autocomplete(request: QueryRequest):
    query = request.query
    top_k = request.top_k
    query_vector = model.encode(query).tolist()
    
    lexical_query = {
        "multi_match": {
            "query": query,
            "fields": ["title^3", "content"],
            "type": "bool_prefix"
        }
    }

    semantic_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'title_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }

    combined_query = {
        "bool": {
            "should": [
                lexical_query,
                semantic_query
            ]
        }
    }

    response = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": combined_query
        }
    )
    
    results = [hit["_source"]["title"] for hit in response["hits"]["hits"]]
    return {"query": query, "results": results}

@app.get("/")
def read_root():
    return {"message": "Hybrid Autocomplete API is running!"}