from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path
from functools import lru_cache  # added for map feature
import requests  # added for map feature
import os

load_dotenv()
GROQ_API_KEY ="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"
#GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = os.getenv("ES_INDEX", "news_reuters_docs")

# Initialize clients
es = Elasticsearch(ES_HOST, request_timeout=60)
model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)

app = FastAPI(title="Smart News Search & Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage
sessions = {}

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class GeoLocation(BaseModel):
    lat: float
    lon: float

class TemporalFilter(BaseModel):
    start: str
    end: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    user_location: Optional[GeoLocation] = None
    temporal_filter: Optional[TemporalFilter] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    session_id: str
    use_memory: bool = True
    user_location: Optional[GeoLocation] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    query_rewritten: Optional[str] = None
    documents_used: List[str] = []

# ============================================================================
# PROMPTS
# ============================================================================

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

IMPORTANT: The documents below are ranked by relevance - the FIRST document is the MOST relevant to the user's question.

Answer the user's question using ONLY the provided documents. Focus primarily on the most relevant documents (especially the first few).

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

# added for map feature - summarizes news at a location
GEO_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a news analyst. Summarize briefly what is happening in the given location "
               "based ONLY on the provided news content. Use 1-2 concise sentences."),
    ("human", "Location: {location}\n\nArticles:\n{articles}\n\nSummary:")
])

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_docs(docs):
    """Format documents for the LLM context"""
    formatted = []
    for i, d in enumerate(docs, 1):
        ranking_marker = ""
        if i == 1:
            ranking_marker = " [MOST RELEVANT]"
        elif i <= 3:
            ranking_marker = " [HIGHLY RELEVANT]"
        
        formatted.append(
            f"Document {i}{ranking_marker}:\n"
            f"Title: {d['title']}\n"
            f"Date: {d['date']}\n"
            f"Content: {d['content'][:1200]}"
        )
    return "\n\n---\n\n".join(formatted)

# added for map feature - converts place names to coordinates
@lru_cache(maxsize=128)
def geocode_place(place: str):
    """Convert place name to lat/lon using OpenStreetMap"""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place, "format": "json", "limit": 1},
            headers={"User-Agent": "smart-news-ir"},
            timeout=5
        )
        if r.status_code != 200 or not r.json():
            return None
        return {"lat": float(r.json()[0]["lat"]), "lon": float(r.json()[0]["lon"])}
    except:
        return None

# added for map feature - creates summary for each location
def summarize_location(location: str, docs: List[dict]) -> str:
    """Generate AI summary of what's happening at a location"""
    articles_text = "\n\n".join(
        f"- {d['title']} ({d['date']}): {d['content'][:1000]}" for d in docs[:5]
    )
    prompt = GEO_SUMMARY_PROMPT.invoke({"location": location, "articles": articles_text})
    summary = llm.invoke(prompt).content.strip()
    return summary

def hybrid_search(
    query: str, 
    user_location: Optional[Dict[str, float]] = None,
    temporal_filter: Optional[Dict[str, str]] = None,
    top_k: int = 10
):
    """Enhanced hybrid search with fuzzy matching, recency, and location awareness"""
    
    # clean query - remove noise words
    noise_words = ['tell me about', 'what did', 'what', 'who', 'when', 'where', 'why', 'how', 'please', 'can you', 'could you', 'i want to know']
    cleaned_query = query.lower()
    for noise in noise_words:
        cleaned_query = cleaned_query.replace(noise, '')
    cleaned_query = ' '.join(cleaned_query.split())
    
    search_query = cleaned_query if len(cleaned_query) < len(query) * 0.7 else query
    print(f"[SEARCH] Original: '{query}'")
    if search_query != query:
        print(f"[SEARCH] Cleaned:  '{search_query}'")
    
    query_vector = model.encode(search_query).tolist()
    
    query_body = {
        "size": top_k,
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            # exact phrase in content
                            {
                                "match_phrase": {
                                    "content": {
                                        "query": search_query,
                                        "boost": 50.0,
                                        "slop": 3
                                    }
                                }
                            },
                            # phrase in title
                            {
                                "match_phrase": {
                                    "title": {
                                        "query": search_query,
                                        "boost": 75.0,
                                        "slop": 3
                                    }
                                }
                            },
                            # fuzzy match in content
                            {
                                "match": {
                                    "content": {
                                        "query": search_query,
                                        "boost": 2.0,
                                        "fuzziness": "AUTO",
                                        "operator": "and"
                                    }
                                }
                            },
                            # fuzzy match in title
                            {
                                "match": {
                                    "title": {
                                        "query": search_query,
                                        "boost": 3.0,
                                        "fuzziness": "AUTO"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                "functions": [
                    # recency boost for historical data
                    {
                        "gauss": {
                            "date": {
                                "origin": "1987-03-01",
                                "scale": "365d",
                                "offset": "30d",
                                "decay": 0.5
                            }
                        },
                        "weight": 0.5
                    }
                ],
                "score_mode": "sum",
                "boost_mode": "sum"
            }
        },
        "knn": [
            {
                "field": "title_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 100,
                "boost": 1.0
            },
            {
                "field": "content_chunks.vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 100,
                "boost": 0.5
            }
        ]
    }
    
    # add geo boost if user location provided
    if user_location and "lat" in user_location and "lon" in user_location:
        query_body["query"]["function_score"]["functions"].append({
            "gauss": {
                "geo_location": {
                    "origin": {
                        "lat": user_location["lat"],
                        "lon": user_location["lon"]
                    },
                    "scale": "200km",
                    "offset": "50km",
                    "decay": 0.5
                }
            },
            "weight": 1.2
        })
    
    # add time filter if provided
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
    
    try:
        response = es.search(
            index=INDEX_NAME,
            body=query_body,
            request_timeout=30
        )
        
        results = []
        print(f"\n[SEARCH DEBUG] Query: '{query}' | Found: {len(response['hits']['hits'])} docs")
        
        for i, hit in enumerate(response["hits"]["hits"], 1):
            title = hit["_source"]["title"]
            score = hit["_score"]
            content_preview = hit["_source"]["content"][:100].replace('\n', ' ')
            
            print(f"  {i}. {title[:60]}... (score: {score:.2f})")
            print(f"     Content: {content_preview}...")
            
            results.append({
                "title": hit["_source"]["title"],
                "content": hit["_source"]["content"],
                "date": hit["_source"].get("date", ""),
                "score": hit["_score"],
                "geo_location": hit["_source"].get("geo_location"),
                "places": hit["_source"].get("places", []),
                "authors": hit["_source"].get("authors", []),
                "dateline": hit["_source"].get("dateline", ""),
                "geopoints": hit["_source"].get("geopoints", []),
                "temporalExpressions": hit["_source"].get("temporalExpressions", []),
                "georeferences": hit["_source"].get("georeferences", [])
            })
        
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def rewrite_query_with_context(question: str, history: List[dict]) -> str:
    """Rewrite query using conversation history"""
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

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    return {
        "message": "Smart News Search & Chat API",
        "endpoints": {
            "autocomplete": "/autocomplete",
            "search": "/search",
            "chat": "/chat",
            "clear_session": "/clear_session/{session_id}"
        }
    }

@app.post("/autocomplete")
def autocomplete(request: QueryRequest):
    """Autocomplete suggestions after 3 characters"""
    try:
        query = request.query
        top_k = request.top_k
        
        if len(query) < 3:
            return {"query": query, "results": []}
        
        response = es.search(
            index=INDEX_NAME,
            body={
                "size": top_k,
                "query": {
                    "match": {
                        "title": {
                            "query": query,
                            "analyzer": "autocomplete_infix_search"
                        }
                    }
                },
                "_source": ["title"]
            },
            request_timeout=30
        )
        
        results = [hit["_source"]["title"] for hit in response["hits"]["hits"]]
        return {"query": query, "results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_documents(request: QueryRequest):
    """
    Enhanced search with map feature
    Returns documents + geo_points for map visualization
    """
    try:
        query = request.query
        top_k = request.top_k
        user_location = request.user_location.dict() if request.user_location else None
        temporal_filter = request.temporal_filter.dict() if request.temporal_filter else None
        
        docs = hybrid_search(
            query=query,
            user_location=user_location,
            temporal_filter=temporal_filter,
            top_k=top_k
        )
        
        # modified - added map feature: group docs by location
        geo_groups: Dict[str, List[dict]] = {}
        documents = []
        
        for doc in docs:
            # extract place from doc
            place = None
            if doc.get("places") and len(doc["places"]) > 0:
                place = doc["places"][0]
            elif doc.get("dateline"):
                place = doc["dateline"].split(",")[0]
            
            # geocode the place
            geo = geocode_place(place) if place else None
            
            formatted_doc = {
                "id": doc.get("id", ""),
                "score": doc["score"],
                "title": doc["title"],
                "content": doc["content"],
                "date": doc["date"],
                "dateline": doc.get("dateline", ""),
                "authors": doc.get("authors", []),
                "places": doc.get("places", []),
                "topics": doc.get("topics", []),
                "people": doc.get("people", []),
                "orgs": doc.get("orgs", []),
                "companies": doc.get("companies", []),
                "exchanges": doc.get("exchanges", []),
                "temporalExpressions": doc.get("temporalExpressions", []),
                "georeferences": doc.get("georeferences", []),
                "geopoints": doc.get("geopoints", []),
                "geo_location": geo,
                "place": place
            }
            documents.append(formatted_doc)
            
            # group by location for map
            if place and geo:
                geo_groups.setdefault(place, []).append(formatted_doc)
        
        # modified - added map feature: create geo_points with summaries
        geo_points = []
        for place, items in geo_groups.items():
            geo = items[0]["geo_location"]
            latest_date = max((d["date"] for d in items if d["date"]), default="")
            summary = summarize_location(place, items)
            
            geo_points.append({
                "lat": geo["lat"],
                "lon": geo["lon"],
                "label": place,
                "count": len(items),
                "latest_date": latest_date,
                "summary": summary,
                "documents": items
            })
        
        return {
            "query": query,
            "total": len(documents),
            "documents": documents,
            "geo_points": geo_points  # added for map feature
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat with conversation memory"""
    try:
        session_id = request.session_id
        query = request.query
        use_memory = request.use_memory
        user_location = request.user_location.dict() if request.user_location else None
        
        # initialize session
        if session_id not in sessions:
            sessions[session_id] = {
                "history": [],
                "last_docs": []
            }
        
        session = sessions[session_id]
        
        # detect follow-up questions
        follow_up_keywords = ["that", "it", "those", "these", "this", "summarize", "explain", "tell me more"]
        is_follow_up = any(kw in query.lower() for kw in follow_up_keywords) and len(query.split()) < 15
        
        # rewrite query if needed
        search_query = query
        query_rewritten = None
        if use_memory and session["history"] and is_follow_up:
            search_query = rewrite_query_with_context(query, session["history"])
            query_rewritten = search_query
            print(f"[CHAT] Query rewritten: '{query}' → '{search_query}'")
        
        # retrieve documents
        if is_follow_up and session["last_docs"]:
            print(f"[CHAT] Reusing {len(session['last_docs'])} cached documents")
            docs = session["last_docs"]
        else:
            print(f"[CHAT] Searching for: '{search_query}'")
            docs = hybrid_search(
                query=search_query,
                user_location=user_location,
                top_k=10
            )
            session["last_docs"] = docs
            
            print(f"[CHAT] Retrieved {len(docs)} documents:")
            for i, doc in enumerate(docs[:5], 1):
                print(f"  {i}. {doc['title'][:80]} (score: {doc['score']:.2f})")
        
        if not docs:
            answer = "No relevant documents found in the index."
            if use_memory:
                session["history"].append({"question": query, "answer": answer})
            return ChatResponse(
                answer=answer,
                session_id=session_id,
                documents_used=[]
            )
        
        # format context
        context = format_docs(docs)
        
        print(f"\n[CHAT DEBUG] Context being sent to LLM:")
        print(f"Number of docs: {len(docs)}")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  Doc {i}: {doc['title'][:60]}...")
        print(f"\nFirst 500 chars of context:")
        print(context[:500])
        print("...\n")
        
        # format history
        history_text = ""
        if use_memory and session["history"]:
            history_text = "\n".join([
                f"User: {turn['question']}\nAssistant: {turn['answer']}"
                for turn in session["history"][-3:]
            ])
        
        # generate response
        messages = CHAT_PROMPT.invoke({
            "question": query,
            "context": context,
            "history": history_text
        })
        
        response = llm.invoke(messages)
        answer = response.content
        
        # store in session
        if use_memory:
            session["history"].append({
                "question": query,
                "answer": answer
            })
        
        # keep only last 10 turns
        if len(session["history"]) > 10:
            session["history"] = session["history"][-10:]
        
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            query_rewritten=query_rewritten,
            documents_used=[d["title"] for d in docs[:5]]
        )
    
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.delete("/clear_session/{session_id}")
def clear_session(session_id: str):
    """Clear conversation history"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}

@app.get("/health")
def health_check():
    """Check if service is running"""
    es_status = es.ping()
    return {
        "status": "healthy" if es_status else "unhealthy",
        "elasticsearch": "connected" if es_status else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)