# HackRx 6.0 – LLM-Powered Insurance Policy Q&A API

## Overview
This project is a production-ready, hackathon-compliant API for answering natural language questions about insurance policy PDFs using LLMs, semantic search, and explainable retrieval.

- **Backend:** FastAPI
- **Vector DB:** Pinecone
- **LLM:** GPT-4 (or Cohere/Perplexity for demo)
- **DB:** PostgreSQL (async, for logging/analytics)
- **Deployment:** Ready for Heroku, Vercel, Railway, Render, AWS, etc.
- **API Auth:** Bearer token
- **Document Support:** PDF only (per hackathon rules)

## Features
- Accepts PDF policy URL and array of questions
- Retrieves and matches relevant policy clauses using Pinecone
- Uses LLM for accurate, explainable answers
- Returns only answer strings (no extra metadata)
- Logs all queries and answers for analytics
- Modular, fast, and hackathon-compliant

## API Usage
### Endpoint
```
POST /hackrx/run
```

### Authentication
Header:
```
Authorization: Bearer <api_key>
```

### Request Example
```
POST /hackrx/run
Content-Type: application/json
Authorization: Bearer <api_key>
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "Does this policy cover maternity expenses?"
  ]
}
```

### Response Example
```
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy."
  ]
}
```

## Deployment
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up PostgreSQL:**
   - Update `.env` or set `DATABASE_URL` if needed
   - Run `init_db.py` to initialize tables
3. **Start FastAPI server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. **Deploy to Heroku/Vercel/Railway/Render** (see platform docs)

## Hackathon Compliance
- **PDF only** (no DOCX/email)
- **POST /hackrx/run** endpoint, Bearer token auth
- **Returns only `answers` list** (no rationale/matched_text)
- **Responds in <30s**
- **Ready for HTTPS/public deployment**

## File Structure
- `main.py` – FastAPI app, API logic
- `db.py` – Async PostgreSQL models
- `init_db.py` – DB migration script
- `requirements.txt` – Python dependencies
- `Procfile`, `render.yaml` – Deployment configs
- `.env` – Secrets (not tracked)

## Remove Before Git Push
- `test_api.py` (local testing only)
- `__pycache__/` (Python cache)
- `.env` (never commit secrets)
- Any other local/dev-only files

---

**Good luck at HackRx 6.0!**

For questions, contact the organizing team or check the hackathon platform guide.
