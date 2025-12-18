from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path
import os


#GROQ_API_KEY ="user's_groq_api_key_here"  # Replace with your actual GROQ API key
# Configuration
ES_HOST = "http://localhost:9200"
INDEX_NAME = "news_reuters_docs"


# Initialize clients
es = Elasticsearch(ES_HOST, request_timeout=60)
model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
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
sessions = {}

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    session_id: str
    use_memory: bool = True

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    query_rewritten: Optional[str] = None
    documents_used: List[str] = []

# Prompts
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

# Helper functions
def format_docs(docs):
    """Format documents for the LLM context."""
    return "\n\n".join(
        f"Title: {d['title']}\nDate: {d['date']}\nContent: {d['content'][:1200]}"
        for d in docs
    )

def hybrid_search(query: str, top_k: int = 10):
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
            "title": hit["_source"]["title"],
            "content": hit["_source"]["content"],
            "date": hit["_source"].get("date", ""),
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]

def rewrite_query_with_context(question: str, history: List[dict]) -> str:
    """Rewrite query using conversation history."""
    if not history:
        return question
    
    # Format recent history
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

# API Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Smart News Search & Chat API",
        "endpoints": {
            "autocomplete": "/autocomplete",
            "chat": "/chat",
            "clear_session": "/clear_session/{session_id}"
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
        
        results = [hit["_source"]["title"] for hit in response["hits"]["hits"]]
        return {"query": query, "results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_documents(request: QueryRequest):
    """Full document search endpoint returning complete document details."""
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
                ]
            },
            request_timeout=30
        )
        
        # Return full document details
        documents = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            doc = {
                "id": hit["_id"],
                "score": hit["_score"],
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
                "geo_location": source.get("geo_location")
            }
            documents.append(doc)
        
        return {
            "query": query,
            "total": len(documents),
            "documents": documents
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
        
        # Initialize session if not exists
        if session_id not in sessions:
            sessions[session_id] = {
                "history": [],
                "last_docs": []
            }
        
        session = sessions[session_id]
        
        # Check if follow-up question
        follow_up_keywords = ["that", "it", "those", "these", "this", "summarize", "explain", "tell me more"]
        is_follow_up = any(kw in query.lower() for kw in follow_up_keywords) and len(query.split()) < 15
        
        # Rewrite query if needed
        search_query = query
        query_rewritten = None
        if use_memory and session["history"] and is_follow_up:
            search_query = rewrite_query_with_context(query, session["history"])
            query_rewritten = search_query
        
        # Retrieve documents (reuse if follow-up)
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
        
        # Format context
        context = format_docs(docs)
        
        # Format history
        history_text = ""
        if use_memory and session["history"]:
            history_text = "\n".join([
                f"User: {turn['question']}\nAssistant: {turn['answer']}"
                for turn in session["history"][-3:]
            ])
        
        # Generate response
        messages = CHAT_PROMPT.invoke({
            "question": query,
            "context": context,
            "history": history_text
        })
        
        response = llm.invoke(messages)
        answer = response.content
        
        # Store in session
        if use_memory:
            session["history"].append({
                "question": query,
                "answer": answer
            })
        
        # Keep only last 10 turns
        if len(session["history"]) > 10:
            session["history"] = session["history"][-10:]
        
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            query_rewritten=query_rewritten,
            documents_used=[d["title"] for d in docs[:5]]
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
    """Health check endpoint."""
    es_status = es.ping()
    return {
        "status": "healthy" if es_status else "unhealthy",
        "elasticsearch": "connected" if es_status else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)