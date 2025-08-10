import fitz
import requests
import openai
from pinecone import Pinecone, ServerlessSpec  # v3 import
from fastapi import FastAPI, HTTPException, Depends, Request, status, APIRouter, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import io
import os
from datetime import datetime
import uuid
import hashlib
import re
import asyncio
# (SQLAlchemy and DB imports moved to bottom where SessionLocal/init_db are used)

# Load environment variables from .env if present (for local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize app and API router with prefix
app = FastAPI(title="HackRX LLM API", version="1.0.0")
router = APIRouter(prefix="/api/v1")

# Security
security = HTTPBearer()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "pdf")

# Request/Response Models
class HackRXRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]  # Only answer strings, per hackathon rules.

API_KEY = os.getenv("HACKRX_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # 1536 dims
# Configurable LLM models
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4")
OLD_PINECONE_INDEX = os.getenv("OLD_PINECONE_INDEX", "")

def embedding_dim_for_model(model: str) -> int:
    # Known OpenAI embedding dims
    if model == "text-embedding-3-large":
        return 3072
    # default
    return 1536

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

def extract_text_from_pdf_url(pdf_url: str) -> str:
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()
    pdf_content = io.BytesIO(response.content)
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    text = " ".join(page.get_text().replace("\n", " ") for page in doc)
    doc.close()
    print(f"[DEBUG] First 500 chars of extracted PDF text:\n{text[:500]}")
    return text


# Cohere removed: OpenAI is the sole LLM provider

# Configure OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Initialize Pinecone (v3 style)
def init_pinecone(index_name, api_key, dim):
    pc = Pinecone(api_key=api_key)
    region = os.getenv("PINECONE_REGION", "us-east-1")
    names = pc.list_indexes().names()
    if index_name not in names:
        print(f"Creating index: {index_name} in region {region} with dimension {dim}")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region=region)
        )
        print(f"Index {index_name} created successfully")
    else:
        try:
            desc = pc.describe_index(index_name)
            existing_dim = desc.get("dimension") or desc.get("spec", {}).get("dimension")
            if existing_dim and existing_dim != dim:
                print(f"[ERROR] Pinecone index '{index_name}' dimension {existing_dim} != required {dim}. Change PINECONE_INDEX or recreate index.")
        except Exception as e:
            print(f"[WARN] Could not describe Pinecone index: {e}")
    return pc.Index(index_name)

try:
    index = init_pinecone(INDEX_NAME, PINECONE_API_KEY, embedding_dim_for_model(EMBEDDING_MODEL))
    print(f"Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"Pinecone connection error: {e}")
    index = None

def ask_openai(query, context_chunks):
    # Use top-5 most relevant chunks, up to 120 words each
    context = "\n\n".join([" ".join(chunk.split()[:120]) for chunk in context_chunks[:5]])
    prompt = (
        "You are an expert assistant. Only answer using the provided PDF context. "
        "If the answer is not in the context, say 'Not found in the document.' "
        "Answer concisely and accurately, in one sentence.\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )
    try:
        messages = [
            {"role": "system", "content": "You answer strictly from the PDF context, never guessing, never citing, never inventing."},
            {"role": "user", "content": prompt}
        ]
        resp, used_model = chat_completion_with_fallback(
            [ANSWER_MODEL, "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            messages,
            max_tokens=80,
            temperature=0.0,
            timeout=20,
        )
        print(f"[DEBUG] Answer generated with model: {used_model}")
        answer = resp["choices"][0]["message"]["content"].strip()
        if not answer or answer.lower().startswith("not found"):
            print("[DEBUG] OpenAI returned no confident answer.")
        return answer
    except Exception as e:
        print("[ERROR] OpenAI API error:", e)
        return "Error: Unable to generate answer"

import re

def chunk_text_by_section(text):
    """
    Splits text into paragraphs/sections, detects headings (e.g. Section, Clause, Policy) for better semantic coherence.
    Returns a list of (section_title, section_text) tuples.
    """
    # Simple regex for headings (customize as needed)
    heading_regex = re.compile(r'(Section|Clause|Policy|\d+\.|[A-Z][a-z]+\s?)+(?=\s*[:.-])', re.MULTILINE)
    lines = text.split('\n')
    chunks = []
    current_title = ""
    current_chunk = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if heading_regex.match(line):
            if current_chunk:
                chunks.append((current_title, " ".join(current_chunk)))
                current_chunk = []
            current_title = line
        else:
            current_chunk.append(line)
    if current_chunk:
        chunks.append((current_title, " ".join(current_chunk)))
    return chunks


# Removed legacy Pinecone initializer (used wrong dimension)

import re

def fallback_chunk_text(text, chunk_size=180, overlap=60):
    """
    Overlapping sentence-grouped chunks for better boundary context.
    chunk_size/overlap measured in words.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    words = []
    for s in sentences:
        words.extend(s.split())
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(n, start + chunk_size)
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(' '.join(chunk_words))
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks

# --- Embedding and selection utilities (OpenAI-based, runtime only) ---
def get_openai_embedding(text: str, model: str = "text-embedding-3-small"):
    """Get a single embedding with fallback to smaller model if needed."""
    candidates = [model]
    if model != "text-embedding-3-small":
        candidates.append("text-embedding-3-small")
    for m in candidates:
        try:
            resp = openai.Embedding.create(model=m, input=text)
            return resp["data"][0]["embedding"]
        except Exception as e:
            print(f"[WARN] Embedding failed for {m}: {e}")
            continue
    print("[ERROR] All embedding fallbacks failed")
    return None

def embed_texts(texts: list, model: str = "text-embedding-3-small"):
    """Batch embeddings with fallback to smaller model if primary fails."""
    candidates = [model]
    if model != "text-embedding-3-small":
        candidates.append("text-embedding-3-small")
    for m in candidates:
        try:
            resp = openai.Embedding.create(model=m, input=texts)
            return [d["embedding"] for d in resp["data"]]
        except Exception as e:
            print(f"[WARN] Batch Embedding failed for {m}: {e}")
            continue
    print("[ERROR] All batch embedding fallbacks failed")
    return [None] * len(texts)

# --- ChatCompletion fallback utility ---
def chat_completion_with_fallback(model_candidates: list, messages: list, **kwargs):
    for m in model_candidates:
        try:
            resp = openai.ChatCompletion.create(model=m, messages=messages, **kwargs)
            return resp, m
        except Exception as e:
            print(f"[WARN] ChatCompletion failed for {m}: {e}")
            continue
    raise RuntimeError("All ChatCompletion fallbacks failed")

def cosine(u, v):
    import math
    if u is None or v is None:
        return -1.0
    dot = sum(a*b for a, b in zip(u, v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    if nu == 0 or nv == 0:
        return -1.0
    return dot / (nu * nv)

def mmr_select(query_vec, doc_vecs, k=8, lambda_coef=0.7):
    selected = []
    candidates = list(range(len(doc_vecs)))
    while candidates and len(selected) < k:
        best_idx = None
        best_score = -1e9
        for i in candidates:
            rel = cosine(query_vec, doc_vecs[i])
            div = 0.0
            if selected:
                div = max(cosine(doc_vecs[i], doc_vecs[j]) for j in selected)
            score = lambda_coef * rel - (1 - lambda_coef) * div
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected

def rerank_with_openai(question: str, chunk_texts: list, max_return: int = 5):
    """Ask OpenAI to pick the most relevant chunk IDs. Returns list of indices."""
    if not chunk_texts:
        return []
    items = []
    for i, t in enumerate(chunk_texts):
        # keep each chunk concise
        items.append(f"ID:{i} -> {t[:500]}")
    rerank_prompt = (
        "You will be given a question and a list of chunked passages with IDs. "
        "Select the top passages most relevant to answering the question. "
        "Return only a comma-separated list of IDs in descending order of relevance, no extra text.\n\n"
        f"Question: {question}\nPassages:\n" + "\n".join(items)
    )
    try:
        messages = [
            {"role": "system", "content": "You are a ranking assistant."},
            {"role": "user", "content": rerank_prompt}
        ]
        resp, used_model = chat_completion_with_fallback(
            [RERANK_MODEL, "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            messages,
            max_tokens=50,
            temperature=0.0,
        )
        print(f"[DEBUG] Rerank generated with model: {used_model}")
        text = resp["choices"][0]["message"]["content"].strip()
        ids = []
        for part in text.replace("\n", ",").split(','):
            part = part.strip()
            if part.startswith("ID:"):
                part = part[3:].strip()
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(chunk_texts):
                    ids.append(idx)
        # unique and cap
        seen = set()
        ordered = []
        for idx in ids:
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        return ordered[:max_return]
    except Exception as e:
        print(f"[ERROR] OpenAI rerank error: {e}")
        return list(range(min(max_return, len(chunk_texts))))

def process_questions_with_model(document_text: str, questions: List[str]) -> List[dict]:
    """
    Embeds document by section/paragraph, persists embeddings by doc hash, retrieves top-3 relevant chunks per query,
    and returns structured answers with matched text and rationale.
    Adds debug logging for chunking and Pinecone queries.
    Fallback to sentence/word chunking if section chunking yields <= 1 chunk.
    """
    try:
        if index is None:
            return [{"answer": "Pinecone index not available", "matched_text": "", "rationale": ""}] * len(questions)
        doc_hash = hashlib.sha256(document_text.encode()).hexdigest()[:16]
        section_chunks = chunk_text_by_section(document_text)
        chunk_texts = [f"{title}: {text}" if title else text for title, text in section_chunks]
        print(f"[DEBUG] Number of section chunks: {len(chunk_texts)}")
        if chunk_texts:
            print(f"[DEBUG] First section chunk: {chunk_texts[0][:200]}")
        # Fallback if only one chunk (likely bad chunking)
        if len(chunk_texts) <= 1:
            print("[DEBUG] Fallback: chunking by sentences/words.")
            chunk_texts = fallback_chunk_text(document_text, chunk_size=180)  # Larger chunks for more context
            print(f"[DEBUG] Number of fallback chunks: {len(chunk_texts)}")
            if chunk_texts:
                print(f"[DEBUG] First fallback chunk: {chunk_texts[0][:200]}")
            section_chunks = [("", chunk) for chunk in chunk_texts]
        # Check Pinecone index stats for this namespace
        stats = index.describe_index_stats()
        namespace_stats = stats.get('namespaces', {}).get(doc_hash, {})
        vector_count = namespace_stats.get('vector_count', 0)
        print(f"[DEBUG] Pinecone index stats for namespace '{doc_hash}': {namespace_stats}")
        already_embedded = vector_count > 0
        if not already_embedded:
            embeddings = embed_texts(chunk_texts, model=EMBEDDING_MODEL)
            if not embeddings or all(v is None for v in embeddings):
                print("[ERROR] Embedding failed for all chunks; aborting upsert.")
                return [{"answer": "Embedding failed; check OPENAI_API_KEY and model/quotas", "matched_text": "", "rationale": ""}] * len(questions)
            print(f"[DEBUG] Embeddings generated for {len(embeddings)} chunks (non-None: {sum(1 for v in embeddings if v is not None)})")
            print(f"[DEBUG] Using namespace for upsert: {doc_hash}")
            pinecone_vectors = []
            for i, vec in enumerate(embeddings):
                if vec is None:
                    continue
                pinecone_vectors.append((f"{doc_hash}-{i}", vec, {"text": chunk_texts[i], "section": section_chunks[i][0]}))
            if not pinecone_vectors:
                print("[ERROR] No valid vectors to upsert after filtering.")
                return [{"answer": "No valid vectors to upsert; embeddings failed", "matched_text": "", "rationale": ""}] * len(questions)
            upsert_response = index.upsert(vectors=pinecone_vectors, namespace=doc_hash)
            print(f"[DEBUG] Pinecone upsert response: {upsert_response}")
            stats = index.describe_index_stats()
            print(f"[DEBUG] Pinecone index stats after upsert: {stats}")
            print(f"[DEBUG] Upserted {len(pinecone_vectors)} vectors to Pinecone namespace {doc_hash}")
        answers = []
        for i, question in enumerate(questions):
            try:
                print(f"[DEBUG] Querying Pinecone for question: {question}")
                query_vec = get_openai_embedding(question, model=EMBEDDING_MODEL)
                if query_vec is None:
                    print("[ERROR] Query embedding failed; returning fallback answer.")
                    answers.append({"answer": "Unable to embed query; please retry", "matched_text": "", "rationale": ""})
                    continue
                print(f"[DEBUG] Query vector dimension: {len(query_vec)}. Using namespace: {doc_hash}")
                results = index.query(
                    vector=query_vec, 
                    top_k=20,  # Retrieve more candidates for MMR and rerank
                    include_metadata=True,
                    namespace=doc_hash
                )
                if i == 0:
                    print(f"[DEBUG] Pinecone results for first question: {results}")
                # Candidate chunks
                candidate_texts = [m['metadata']['text'] for m in results['matches']]
                matched_sections = [m['metadata'].get('section', '') for m in results['matches']]
                # Embed query and candidates with OpenAI for MMR selection
                oq = get_openai_embedding(question)
                odocs = embed_texts(candidate_texts)
                # filter out None embeddings and keep mapping
                filtered = [(j, v) for j, v in enumerate(odocs) if v is not None]
                if oq is not None and filtered:
                    idx_map = [j for j, _ in filtered]
                    vecs = [v for _, v in filtered]
                    mmr_rel_idx = mmr_select(oq, vecs, k=min(8, len(vecs)), lambda_coef=0.7)
                    mmr_idx = [idx_map[j] for j in mmr_rel_idx]
                    mmr_chunks = [candidate_texts[j] for j in mmr_idx]
                else:
                    mmr_chunks = candidate_texts[:8]
                # LLM rerank to choose final top-5
                order = rerank_with_openai(question, mmr_chunks, max_return=5)
                context_chunks = [mmr_chunks[j] for j in order]
                answer = ask_openai(question, context_chunks)
                rationale = f"Matched sections: {matched_sections}. Context used: {context_chunks}"
                answers.append({
                    "answer": answer if answer else "Unable to generate answer",
                    "matched_text": context_chunks,
                    "rationale": rationale
                })
            except Exception as e:
                print(f"[DEBUG] Error during Pinecone/LLM for question: {e}")
                answers.append({"answer": f"Error processing question: {str(e)}", "matched_text": "", "rationale": ""})
        return answers
    except Exception as e:
        print(f"[DEBUG] Error processing document: {e}")
        return [{"answer": f"Error processing document: {str(e)}", "matched_text": "", "rationale": ""}] * len(questions)


from db import SessionLocal, QueryLog, init_db
import asyncio

@app.on_event("startup")
async def on_startup():
    await init_db()

@router.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Avoid logging secrets or full Authorization headers in production
    if credentials.scheme.lower() != "bearer" or credentials.credentials != API_KEY:
        print("[DEBUG] Authorization failed!")
        raise HTTPException(status_code=401, detail="Unauthorized")
    document_url = request.documents
    questions = request.questions
    document_text = extract_text_from_pdf_url(document_url)
    answers_full = process_questions_with_model(document_text, questions)
    # Only return answer strings per hackathon rules
    answers = [a["answer"] if isinstance(a, dict) else str(a) for a in answers_full]
    # Log to DB (keep full info for analytics)
    async with SessionLocal() as session:
        for q, a_full in zip(questions, answers_full):
            log = QueryLog(document_url=document_url, question=q, answer=a_full.get("answer", ""), matched_text=str(a_full.get("matched_text", "")), rationale=a_full.get("rationale", ""))
            session.add(log)
        await session.commit()
    return HackRXResponse(answers=answers)

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/")
async def root():
    return {"message": "HackRX LLM API is running"}

# Admin endpoint: delete an old Pinecone index (use with caution). Protected by same Bearer token.
@router.post("/admin/pinecone/delete-index")
async def delete_old_index(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme.lower() != "bearer" or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not OLD_PINECONE_INDEX:
        raise HTTPException(status_code=400, detail="OLD_PINECONE_INDEX not set")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        names = pc.list_indexes().names()
        if OLD_PINECONE_INDEX in names:
            pc.delete_index(OLD_PINECONE_INDEX)
            return {"deleted": OLD_PINECONE_INDEX}
        else:
            return {"message": f"Index '{OLD_PINECONE_INDEX}' not found", "available": names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete index: {e}")

# Include router
app.include_router(router)

# Uvicorn runner for local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
