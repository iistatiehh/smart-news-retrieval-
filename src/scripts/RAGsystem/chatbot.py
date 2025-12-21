  
# %pip install langchain langchain-community langchain-groq elasticsearch sentence-transformers
# %pip install langchain.schema


  
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


  
from elasticsearch import Elasticsearch

ES_INDEX = "news_reuters_docs"

es = Elasticsearch(
    "http://localhost:9200",
    request_timeout=60
)

assert es.ping(), "Elasticsearch not reachable"


  
from typing import List, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer
from pydantic import Field, PrivateAttr

class ElasticsearchHybridRetriever(BaseRetriever):
    es: Any = Field(...)
    index_name: str = Field(...)
    k: int = Field(default=10)

    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self._model.encode(query).tolist()

        body = {
            "size": self.k,
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
                    "k": self.k,
                    "num_candidates": 100
                },
                {
                    "field": "content_chunks.vector",
                    "query_vector": query_vector,
                    "k": self.k,
                    "num_candidates": 100
                }
            ]
        }

        response = self.es.search(index=self.index_name, body=body)

        docs = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            docs.append(
                Document(
                    page_content=src.get("content", ""),
                    metadata={
                        "title": src.get("title"),
                        "date": src.get("date"),
                        "score": hit["_score"]
                    }
                )
            )
        return docs

  
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)

# Prompt for query rewriting with context
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

# Main system prompt
SYSTEM_PROMPT = """
You are a retrieval-augmented assistant.
Answer the user's question using ONLY the provided documents and conversation history.
If the answer is not present in the documents, say clearly that the information is not available.
Always cite relevant documents by their title and date.
After answering, list the most useful documents and explain briefly why each one is useful.

When the user refers to previous topics (like "that one", "it", "the previous topic"), 
use the conversation history to understand what they're referring to.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", """
Conversation History:
{history}

Question:
{question}

Documents:
{context}
""")
    ]
)

  
def format_docs(docs):
    return "\n\n".join(
        f"{d.metadata['title']} ({d.metadata['date']})\n{d.page_content[:1200]}"
        for d in docs
    )

  
# Conversation memory storage
conversation_history = []
last_retrieved_docs = []  # Store documents from the last retrieval

retriever = ElasticsearchHybridRetriever(
    es=es,
    index_name="news_reuters_docs",
    k=10
)

def rewrite_query_with_context(question: str) -> str:
    """
    Rewrite the query to include context from conversation history.
    This helps with follow-up questions like "summarize that one".
    """
    if not conversation_history:
        return question
    
    # Format recent history
    history = "\n".join([
        f"User: {turn['question']}\nAssistant: {turn['answer'][:300]}..."
        for turn in conversation_history[-2:]  # Last 2 turns
    ])
    
    # Use LLM to rewrite query
    messages = QUERY_REWRITE_PROMPT.invoke({
        "history": history,
        "question": question
    })
    
    rewritten = llm.invoke(messages).content.strip()
    return rewritten

def chatbot(question: str, use_memory: bool = True, reuse_docs: bool = True):
    """
    Main chatbot function with memory support.
    
    Args:
        question: User's question
        use_memory: Whether to use conversation history (default: True)
        reuse_docs: Whether to reuse documents from previous turn for follow-ups (default: True)
    """
    global last_retrieved_docs
    
    # Check if this is a follow-up question that references previous content
    follow_up_keywords = ["that", "it", "those", "these", "this", "summarize", "explain", "tell me more"]
    is_follow_up = any(keyword in question.lower() for keyword in follow_up_keywords) and len(question.split()) < 15
    
    # Rewrite query with context if it's a follow-up
    search_query = question
    if use_memory and conversation_history and is_follow_up:
        search_query = rewrite_query_with_context(question)
        print(f"[Query Rewrite] Original: '{question}' → Rewritten: '{search_query}'")
    
    # Decide whether to retrieve new documents or reuse previous ones
    if reuse_docs and is_follow_up and last_retrieved_docs:
        print(f"[Using cached documents from previous turn]")
        docs = last_retrieved_docs
    else:
        # Retrieve relevant documents
        docs = retriever._get_relevant_documents(search_query)
        last_retrieved_docs = docs  # Cache for potential reuse

    if not docs:
        response_text = "No relevant documents found in the index."
        if use_memory:
            conversation_history.append({"question": question, "answer": response_text})
        return response_text

    # Format documents
    context = format_docs(docs)
    
    # Format conversation history
    history = ""
    if use_memory and conversation_history:
        history = "\n".join([
            f"User: {turn['question']}\nAssistant: {turn['answer']}"
            for turn in conversation_history[-3:]  # Keep last 3 turns
        ])

    # Generate response
    messages = prompt.invoke({
        "question": question,
        "context": context,
        "history": history
    })

    response = llm.invoke(messages)
    response_text = response.content
    
    # Store in memory
    if use_memory:
        conversation_history.append({
            "question": question,
            "answer": response_text
        })
    
    return response_text

def clear_memory():
    """Clear conversation history and cached documents."""
    global conversation_history, last_retrieved_docs
    conversation_history = []
    last_retrieved_docs = []
    print("Conversation memory cleared.")

def show_memory():
    """Display current conversation history."""
    if not conversation_history:
        print("No conversation history.")
        return
    
    print("\n=== Conversation History ===")
    for i, turn in enumerate(conversation_history, 1):
        print(f"\nTurn {i}:")
        print(f"Q: {turn['question']}")
        print(f"A: {turn['answer'][:200]}...")  # Show first 200 chars
    print("=" * 30)

  
print("Turn 1:")
print(chatbot("tell me about NATIONAL AVERAGE PRICES FOR FARMER-OWNED RESERVE"))
print("\n" + "="*80 + "\n")

print("Turn 2 (referencing previous answer):")
print(chatbot("Summarize that one in two sentences"))
print("\n" + "="*80 + "\n")

# Show conversation history
show_memory()

  
# Example: Start a new conversation
clear_memory()

question = "What economic events affected Japan in the late 1990s?"
answer = chatbot(question)
print(answer)

  
# Continue the conversation
follow_up = "What were the consequences of those events?"
answer = chatbot(follow_up)
print(answer)
