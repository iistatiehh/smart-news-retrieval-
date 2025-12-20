from dotenv import load_dotenv
import os

load_dotenv()

from functools import lru_cache
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# =============================
# Configuration
# =============================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9202")
INDEX_NAME = os.getenv("ES_INDEX", "news_reuters_docs")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # لازم يكون موجود بملف .env


# =============================
# Initialize clients
# =============================
es = Elasticsearch(ES_HOST, request_timeout=60)
model = SentenceTransformer("all-MiniLM-L6-v2")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)

app = FastAPI(title="Smart News Search & Chat API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis for production)
sessions: Dict[str, Dict[str, Any]] = {}


# =============================
# Pydantic models
# =============================
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class ChatRequest(BaseModel):
    query: str
    session_id: str
    use_memory: bool = True


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    query_rewritten: Optional[str] = None
    documents_used: List[str] = []


# =============================
# Prompts
# =============================
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter. Given a conversation history and a follow-up question,
rewrite the follow-up question to be a standalone question that includes necessary context.

If the question references previous topics (like "that", "it", "those events", "summarize that one"),
incorporate the relevant topic from the conversation history into the rewritten query.

If the question is already standalone, return it as-is."""),
    ("human", """
Conversation History:
{history}

Follow-up Question: {question}

Rewritten Query (return ONLY the rewritten query, no explanations):""")
])

SYSTEM_PROMPT = """
You are a retrieval-augmented assistant for Reuters news articles.
Answer the user's question using ONLY the provided documents and conversation history.
If the answer is not present in the documents, say clearly that the information is not available.
Always cite relevant documents by their title and date.
Keep your answers concise and informative.
"""

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """
Conversation History:
{history}

Question:
{question}

Documents:
{context}
""")
])


# =============================
# Helper functions
# =============================

def format_docs(docs: List[dict]) -> str:
    """Format documents for the LLM context."""
    return "\n\n".join(
        f"Title: {d['title']}\nDate: {d['date']}\nContent: {d['content'][:1200]}"
        for d in docs
    )


@lru_cache(maxsize=128)
def geocode_place(place: str):
    """
    Convert place name (TOKYO, JAPAN, etc.) to lat/lon using OpenStreetMap.
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": place,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "smart-news-ir-project"
        }

        r = requests.get(url, params=params, headers=headers, timeout=5)
        if r.status_code != 200:
            return None

        data = r.json()
        if not data:
            return None

        return {
            "lat": float(data[0]["lat"]),
            "lon": float(data[0]["lon"])
        }

    except Exception:
        return None



def hybrid_search(query: str, top_k: int = 10) -> List[dict]:
    """Perform hybrid search combining lexical and semantic search."""
    query_vector = model.encode(query).tolist()

    response = es.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "content"]
                }
            },
            "knn": [
                {
                    "field": "title_vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": 100
                },
                {
                    "field": "content_chunks.vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": 100
                }
            ]
        },
        request_timeout=30
    )

    return [
        {
            "title": hit["_source"].get("title", ""),
            "content": hit["_source"].get("content", ""),
            "date": hit["_source"].get("date", ""),
            "score": hit.get("_score")
        }
        for hit in response.get("hits", {}).get("hits", [])
    ]


def rewrite_query_with_context(question: str, history: List[dict]) -> str:
    """Rewrite query using conversation history."""
    if not history:
        return question

    history_text = "\n".join([
        f"User: {turn['question']}\nAssistant: {turn['answer'][:300]}..."
        for turn in history[-2:]
    ])

    messages = QUERY_REWRITE_PROMPT.invoke({
        "history": history_text,
        "question": question
    })

    rewritten = llm.invoke(messages).content.strip()
    return rewritten


# =============================
# API Endpoints
# =============================
@app.get("/")
def read_root():
    return {
        "message": "Smart News Search & Chat API",
        "endpoints": {
            "autocomplete": "/autocomplete",
            "search": "/search",
            "chat": "/chat",
            "clear_session": "/clear_session/{session_id}",
            "health": "/health"
        }
    }


@app.post("/autocomplete")
def autocomplete(request: QueryRequest):
    """Autocomplete endpoint for search suggestions."""
    try:
        query = request.query
        top_k = request.top_k
        query_vector = model.encode(query).tolist()

        response = es.search(
            index=INDEX_NAME,
            body={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "content"],
                        "type": "bool_prefix"
                    }
                },
                "knn": {
                    "field": "title_vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": 100
                }
            },
            request_timeout=30
        )

        results = [hit["_source"].get("title", "") for hit in response.get("hits", {}).get("hits", [])]
        return {"query": query, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_documents(request: QueryRequest):
    """Full document search endpoint returning complete document details + geo_summary (SAFE)."""
    try:
        query = request.query
        top_k = request.top_k
        query_vector = model.encode(query).tolist()

        response = es.search(
            index=INDEX_NAME,
            body={
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "content"]
                    }
                },
                "knn": [
                    {
                        "field": "title_vector",
                        "query_vector": query_vector,
                        "k": top_k,
                        "num_candidates": 100
                    },
                    {
                        "field": "content_chunks.vector",
                        "query_vector": query_vector,
                        "k": top_k,
                        "num_candidates": 100
                    }
                ],
                # ✅ Geo Aggregation (does NOT affect search hits)
                "aggs": {
                    "geo_hotspots": {
                        "geotile_grid": {
                            "field": "geo_location",
                            "precision": 3
                        },
                        "aggs": {
                            "center": {
                                "geo_centroid": {
                                    "field": "geo_location"
                                }
                            }
                        }
                    }
                }
            },
            request_timeout=30
        )

        # =========================
        # Documents (OLD – unchanged)
        # =========================
        documents = []

        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})

            # --------- Extract place name ----------
            place_name = None
            if source.get("places"):
                place_name = source["places"][0]
            elif source.get("dateline"):
                place_name = source["dateline"].split(",")[0]

            # --------- Geocode ----------
            geo_location = None
            if place_name:
                geo_location = geocode_place(place_name)

            doc = {
                "id": hit.get("_id", ""),
                "score": hit.get("_score"),
                "title": source.get("title", ""),
                "content": source.get("content", ""),
                "date": source.get("date", ""),
                "dateline": source.get("dateline", ""),
                "authors": source.get("authors", []),
                "places": source.get("places", []),
                "topics": source.get("topics", []),
                "people": source.get("people", []),
                "orgs": source.get("orgs", []),
                "companies": source.get("companies", []),
                "exchanges": source.get("exchanges", []),
                "temporalExpressions": source.get("temporalExpressions", []),
                "georeferences": source.get("georeferences", []),
                "geopoints": source.get("geopoints", []),
                "geo_location": geo_location
            }

            documents.append(doc)


        # =========================
        # NEW: Geo Summary (SAFE)
        # =========================
        geo_summary = []
        aggs = response.get("aggregations", {})
        buckets = aggs.get("geo_hotspots", {}).get("buckets", [])

        for b in buckets:
            # ممكن center/ location تكون null
            center_obj = b.get("center", {})
            center = center_obj.get("location") if center_obj else None
            if center and "lat" in center and "lon" in center:
                geo_summary.append({
                    "lat": center["lat"],
                    "lon": center["lon"],
                    "count": b.get("doc_count", 0)
                })

        return {
            "query": query,
            "total": len(documents),
            "documents": documents,
            "geo_summary": geo_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat endpoint with memory and context."""
    try:
        session_id = request.session_id
        query = request.query
        use_memory = request.use_memory

        if session_id not in sessions:
            sessions[session_id] = {"history": [], "last_docs": []}

        session = sessions[session_id]

        follow_up_keywords = ["that", "it", "those", "these", "this", "summarize", "explain", "tell me more"]
        is_follow_up = any(kw in query.lower() for kw in follow_up_keywords) and len(query.split()) < 15

        search_query = query
        query_rewritten = None
        if use_memory and session["history"] and is_follow_up:
            search_query = rewrite_query_with_context(query, session["history"])
            query_rewritten = search_query

        if is_follow_up and session["last_docs"]:
            docs = session["last_docs"]
        else:
            docs = hybrid_search(search_query, top_k=10)
            session["last_docs"] = docs

        if not docs:
            answer = "No relevant documents found in the index."
            if use_memory:
                session["history"].append({"question": query, "answer": answer})
            return ChatResponse(
                answer=answer,
                session_id=session_id,
                documents_used=[]
            )

        context = format_docs(docs)

        history_text = ""
        if use_memory and session["history"]:
            history_text = "\n".join([
                f"User: {turn['question']}\nAssistant: {turn['answer']}"
                for turn in session["history"][-3:]
            ])

        messages = CHAT_PROMPT.invoke({
            "question": query,
            "context": context,
            "history": history_text
        })

        answer = llm.invoke(messages).content

        if use_memory:
            session["history"].append({"question": query, "answer": answer})

        if len(session["history"]) > 10:
            session["history"] = session["history"][-10:]

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            query_rewritten=query_rewritten,
            documents_used=[d.get("title", "") for d in docs[:5]]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear_session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history for a session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}


@app.get("/health")
def health_check():
    es_status = es.ping()
    return {
        "status": "healthy" if es_status else "unhealthy",
        "elasticsearch": "connected" if es_status else "disconnected",
        "index": INDEX_NAME,
        "es_host": ES_HOST
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
