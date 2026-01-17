import streamlit as st
import requests
import uuid

# ---------------------------------------------------------
# CONFIGURATION & THEME
# ---------------------------------------------------------
API_BASE_URL = "http://127.0.0.1:8000"
MAX_MESSAGES_TO_STORE = 30  # Safety limit for browser memory

st.set_page_config(
    page_title="Enterprise Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Adaptive CSS
st.markdown("""
<style>
    [data-testid="stChatMessage"] {
        max-width: 850px;
        margin: auto;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--secondary-background-color);
    }
    div.stButton > button {
        width: 100%;
        text-align: left;
        background-color: transparent;
        color: var(--text-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 6px;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    div.stButton > button:hover {
        background-color: var(--secondary-background-color);
        border-color: var(--primary-color);
    }
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border-bottom: 1px solid var(--secondary-background-color);
        border-radius: 0;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: var(--secondary-background-color) !important;
    }
    [data-testid="stChatInput"] {
        max-width: 850px;
        margin: auto;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION MANAGEMENT
# -----------------------------
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}

if "active_session_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.all_chats[new_id] = {"title": "New Session", "messages": []}
    st.session_state.active_session_id = new_id

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.all_chats[new_id] = {"title": "New Session", "messages": []}
    st.session_state.active_session_id = new_id

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### Control Panel")
    if st.button("＋ New Session", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### History")
    
    chat_ids = list(st.session_state.all_chats.keys())
    for c_id in reversed(chat_ids):
        chat_info = st.session_state.all_chats[c_id]
        is_active = (c_id == st.session_state.active_session_id)
        label = f"● {chat_info['title']}" if is_active else f"○ {chat_info['title']}"
        
        if st.button(label, key=c_id):
            st.session_state.active_session_id = c_id
            st.rerun()

    st.markdown("---")
    with st.expander("📁 Knowledge Base", expanded=False):
        uploaded_files = st.file_uploader("Upload", accept_multiple_files=True, type=["pdf", "docx", "txt", "csv"], label_visibility="collapsed")
        if uploaded_files:
            for f in uploaded_files:
                if f.name not in st.session_state.processed_files:
                    with st.spinner(f"Indexing {f.name}..."):
                        try:
                            # SEND FILE TO BACKEND
                            files = {"file": (f.name, f.getvalue(), f.type)}
                            res = requests.post(f"{API_BASE_URL}/upload_document", files=files)
                            if res.status_code == 200:
                                st.session_state.processed_files.add(f.name)
                                st.toast(f"Successfully Indexed: {f.name}")
                            else:
                                st.error(f"Failed to upload {f.name}")
                        except Exception as e:
                            st.error(f"Upload error: {e}")

# -----------------------------
# MAIN INTERFACE
# -----------------------------
active_chat = st.session_state.all_chats[st.session_state.active_session_id]
USER_AVATAR = "👤"
AI_AVATAR = "🏢"

st.markdown("<h3 style='text-align: center; opacity: 0.7;'>Intelligence Portal</h3>", unsafe_allow_html=True)

# 1. Message Display
if not active_chat["messages"]:
    st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.5;'>System ready. Inquire about your knowledge base.</p>", unsafe_allow_html=True)
else:
    for message in active_chat["messages"]:
        avatar = USER_AVATAR if message["role"] == "user" else AI_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# 2. Input Logic
if prompt := st.chat_input("Enter your query..."):
    # Add User Message
    active_chat["messages"].append({"role": "user", "content": prompt})
    
    # Auto-update session title
    if active_chat["title"] == "New Session":
        active_chat["title"] = prompt[:25] + ("..." if len(prompt) > 25 else "")
    
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Generate Assistant Response
    with st.chat_message("assistant", avatar=AI_AVATAR):
        placeholder = st.empty()
        try:
            with st.spinner("Consulting knowledge base..."):
                response = requests.post(
                    f"{API_BASE_URL}/query", 
                    json={
                        "query": prompt,
                        "chat_history": active_chat["messages"]
                    }
                )
                if response.status_code == 200:
                    answer = response.json().get("response", "No response.")
                    placeholder.markdown(answer)
                    active_chat["messages"].append({"role": "assistant", "content": answer})
                    
                    # --- FRONTEND HISTORY LIMIT ---
                    # Ensure we don't store more than X messages in browser state
                    if len(active_chat["messages"]) > MAX_MESSAGES_TO_STORE:
                        active_chat["messages"] = active_chat["messages"][-MAX_MESSAGES_TO_STORE:]
                        
                    st.rerun()
                else:
                    placeholder.error("Service error. Please verify backend status.")
        except Exception as e:
            placeholder.error(f"Connection failure: {e}")
