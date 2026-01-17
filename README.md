# fastapi_server_tutorial

Project contains a FastAPI backend and a Streamlit frontend.

Run locally

- Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app2:app --reload --host 0.0.0.0 --port 8000
```

- Frontend (Streamlit):

```bash
cd frontend
pip install -r requirements.txt
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
```

Deployment notes

- Backend service root: `backend` — start with `uvicorn app2:app --host 0.0.0.0 --port $PORT`.
- Frontend service root: `frontend` — start with `streamlit run frontend.py --server.port $PORT --server.address 0.0.0.0`.
- Do NOT commit `backend/.env`; instead set `GROQ_API_KEY` in the host's environment/secret settings.
- Chroma uses `/tmp/chroma_db` (ephemeral on many free hosts).
