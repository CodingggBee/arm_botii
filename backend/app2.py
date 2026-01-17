import io
import os
import pandas as pd
from typing import Annotated, List, Set, Dict, Any
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from docx import Document as DocxDocument

# --- LangChain Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.tools import Tool
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory # <--- NEW: Memory Import


# -----------------------------
# DATA MODELS (NEW SECTION)
# -----------------------------
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: List[Message] = []
# -----------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Global Set to track filenames
uploaded_filenames: Set[str] = set()

# Initialize Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Vector DB
vector_store = Chroma(
    # persist_directory="./chroma_db",
    persist_directory="/tmp/chroma_db",
    embedding_function=embedding_model
)
try:
    # Query Chroma for all stored data
    existing_data = vector_store.get()
    if existing_data and existing_data['metadatas']:
        for meta in existing_data['metadatas']:
            if meta and 'source' in meta:
                uploaded_filenames.add(meta['source'])
    print(f"✅ Recovery Complete: Loaded {len(uploaded_filenames)} files from disk.")
except Exception as e:
    print(f"⚠️ Initial sync failed: {e}")

# Initialize LLM (Groq)
groq_api_key = os.getenv("GROQ_API_KEY", None)

llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=groq_api_key
)

# -----------------------------
# 2. AGENT MEMORY & TOOLS
# -----------------------------

# --- GLOBAL MEMORY ---
# This allows the agent to remember context (e.g., "Summarize it").
# Note: In a real production app, you would use a database for this. 
# For this tutorial, a global variable works for a single user.
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# --- TOOL 1: SMART SEARCH ---
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

def search_documents(query: str) -> str:
    try:
        # If query is vague (e.g., "summary", "it"), the agent might pass that directly.
        # We ensure we search for meaningful content.
        search_term = query
        if len(query) < 4 or query.lower() in ["summary", "summarize", "it", "this"]:
            search_term = "overview summary main points"
            
        docs = retriever.invoke(search_term)
        if not docs:
            return "NO_RESULTS_FOUND. Tell the user you couldn't find that specific info in the uploaded docs."
        return "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs])
    except Exception as e:
        return f"Error searching: {str(e)}"

search_tool = Tool(
    name="search_enterprise_documents",
    func=search_documents,
    description="Use this tool ONLY if the user asks a specific question about the uploaded document content."
)

# --- TOOL 2: INVENTORY ---
def list_files(query: str) -> str:
    # Query Chroma for all unique 'source' metadata entries
    try:
        data = vector_store.get()
        if not data or not data['metadatas']:
            return "No documents found in the persistent knowledge base."
        
        # Extract unique filenames from metadata
        sources = {m.get('source') for m in data['metadatas'] if m.get('source')}
        
        if not sources:
            return "No documents found."
            
        return "Current Files in Knowledge Base:\n" + "\n".join(f"- {name}" for name in sources)
    except Exception as e:
        return f"Error accessing inventory: {str(e)}"

list_tool = Tool(
    name="list_uploaded_files",
    func=list_files,
    description="Use this tool to see which documents are currently stored in the system.",
)

tools = [search_tool, list_tool]

# --- FALLBACK HANDLER ---
def fallback_handler(error):
    # This catches the "Agent stopped" errors and "Parsing" errors
    error_str = str(error)
    if "None is not a valid tool" in error_str:
        return "Please answer the user directly using your general knowledge."
    if "Could not parse LLM output: `" in error_str:
        return error_str.split("Could not parse LLM output: `")[1].split("`")[0]
    return "I apologize, but I encountered a temporary error. Please try again."

# --- SYSTEM PROMPT (The "ChatGPT" Personality) ---
system_message = """You are a helpful and intelligent AI Assistant.

BEHAVIOR GUIDELINES:
1. **Chat like a Human:** For greetings ("Hi"), general questions ("What is 2+2?", "Write python code"), or small talk, DO NOT use any tools. Just reply naturally using your own knowledge.
2. **Use Context:** You have a memory. If the user says "Summarize it" or "Explain that", refer to the previous conversation or the uploaded documents.
3. **Document Search:** Only use 'search_enterprise_documents' if the user specifically asks about the *uploaded files*.
4. **Inventory:** Use 'list_uploaded_files' only when asked about file availability.

If a tool returns "NO_RESULTS_FOUND", do not retry. Just apologize and say you couldn't find it in the documents.
"""

# --- INITIALIZE CONVERSATIONAL AGENT ---
# We use CHAT_CONVERSATIONAL_REACT_DESCRIPTION which is optimized for chat history + tools
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, # <--- The Upgrade
    verbose=True,
    memory=memory, # <--- Connects the Memory
    handle_parsing_errors=fallback_handler,
    agent_kwargs={
        "system_message": system_message # <--- Injects our rules
    }
)

# -----------------------------
# 3. FASTAPI APP
# -----------------------------
app = FastAPI(title="Enterprise Agentic RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 4. HELPER FUNCTIONS
# -----------------------------
async def parse_file(file: UploadFile, content: bytes) -> str:
    ext = os.path.splitext(file.filename)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext == ".txt":
            return content.decode("utf-8")
        elif ext == ".csv":
            df = pd.read_csv(io.BytesIO(content))
            return df.to_string()
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(io.BytesIO(content))
            return df.to_string()
        elif ext in [".doc", ".docx"]:
            doc = DocxDocument(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        print(f"Error parsing file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse {ext} file: {str(e)}")

# -----------------------------
# 5. API ENDPOINTS
# -----------------------------

@app.get("/", tags=["System"])
async def root():
    return {"message": "Agentic RAG API is running."}

@app.get("/architecture", tags=["Documentation"]) 
async def get_architecture():
    return {
        "architecture_type": "Agentic RAG (Conversational Memory)",
        "components": {
            "llm": "Llama-3.1-8b (via Groq)",
            "vector_store": "ChromaDB",
            "memory": "ConversationBufferMemory",
            "tools": ["Search", "Inventory"]
        }
    }

@app.post("/upload_document", tags=["Ingestion"]) 
async def upload_document(file: Annotated[UploadFile, File()]):
    content = await file.read()
    text = await parse_file(file, content)
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="The file appears to be empty.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([text])
    
    for chunk in chunks:
        chunk.metadata["source"] = file.filename

    if chunks:
        vector_store.add_documents(chunks)
        # No need to manually manage the 'set' anymore if we query the DB 
        # But we can keep it for quick tracking if desired
        uploaded_filenames.add(file.filename)
        
        # Optional: Force persistence for older Chroma versions
        # vector_store.persist() 
        
    return {"status": "Successfully indexed and saved to disk."}

@app.post("/query", tags=["Agentic Reasoning"]) 
async def query_agent(request: QueryRequest):
    query = request.query
    history = request.chat_history

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # --- LIMIT HISTORY HERE ---
        # We only take the last 6 messages (3 turns of User/AI) 
        # to keep the context window clean and fast.
        MAX_HISTORY_LENGTH = 6 
        limited_history = history[-MAX_HISTORY_LENGTH:]

        # 1. Clear the agent's global memory to avoid mixing sessions
        memory.chat_memory.clear()

        # 2. Reload ONLY the limited history sent from Streamlit
        for msg in limited_history:
            if msg.role == "user":
                memory.chat_memory.add_user_message(msg.content)
            elif msg.role == "assistant":
                memory.chat_memory.add_ai_message(msg.content)

        # 3. Invoke Agent with the limited context
        response = agent_executor.invoke({"input": query})
        
        final_output = response.get("output", "").strip()
        
        return {
            "query": query,
            "response": final_output
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            "query": query, 
            "response": "I encountered an error processing your request."
        }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
app = app