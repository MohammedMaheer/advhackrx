import fitz
import numpy as np
import cohere
import requests
from pinecone import Pinecone, ServerlessSpec  # v3 import
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import io
import os
from datetime import datetime
import uuid

# Initialize app and API router with prefix
app = FastAPI(title="HackRX LLM API", version="1.0.0")
router = APIRouter(prefix="/api/v1")

# Security
security = HTTPBearer()
PINECONE_API_KEY = "pcsk_7B3Z93_8WBKxheRs5H22N8LeMJTCWzjPR1wUZKE8oUJzHDyhMot6qbZ1JrfSkKM7kcLVu7"
INDEX_NAME = "pdf"

# Request/Response Models
class HackRXRequest(BaseModel):
    documents: str  # URL to PDF
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]  # Only answer strings, per hackathon rules.

API_KEY = "bfb8fabaf1ce137c1402366fb3d5a052836234c1ff376c326842f52e3164cc33"

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


# Initialize Cohere Client
co = cohere.Client("ba9VI3VW1sXTxyIKhOZHWPA3326tAQzHGVVQ16aI")

import hashlib
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


# Initialize Pinecone (v3 style)
def init_pinecone(index_name, api_key):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"Index {index_name} created successfully")
    return pc.Index(index_name)

try:
    index = init_pinecone(INDEX_NAME, PINECONE_API_KEY)
    print(f"Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"Pinecone connection error: {e}")
    index = None

def ask_perplexity(query, context_chunks):
    api_key = "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"
    url = "https://api.perplexity.ai/chat/completions"
    short_chunks = [" ".join(chunk.split()[:100]) for chunk in context_chunks]
    context_text = "\n\n".join(short_chunks)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Strictly answer the user's question in only one sentence. Do not provide explanations or extra information and dont cite your answers"},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Error:", response.status_code, response.text)
        return None

import re

def fallback_chunk_text(text, chunk_size=100):
    # Split by sentences, then group into chunks of ~chunk_size words
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = []
    count = 0
    for sent in sentences:
        words = sent.split()
        count += len(words)
        current.append(sent)
        if count >= chunk_size:
            chunks.append(' '.join(current))
            current = []
            count = 0
    if current:
        chunks.append(' '.join(current))
    return chunks

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
            chunk_texts = fallback_chunk_text(document_text)
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
            response = co.embed(
                texts=chunk_texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            embeddings = response.embeddings
            print(f"[DEBUG] Embeddings shape: {len(embeddings)} x {len(embeddings[0]) if embeddings else 0}")
            print(f"[DEBUG] Using namespace for upsert: {doc_hash}")
            pinecone_vectors = [
                (f"{doc_hash}-{i}", vec, {"text": chunk_texts[i], "section": section_chunks[i][0]}) 
                for i, vec in enumerate(embeddings)
            ]
            upsert_response = index.upsert(vectors=pinecone_vectors, namespace=doc_hash)
            print(f"[DEBUG] Pinecone upsert response: {upsert_response}")
            stats = index.describe_index_stats()
            print(f"[DEBUG] Pinecone index stats after upsert: {stats}")
            print(f"[DEBUG] Upserted {len(pinecone_vectors)} vectors to Pinecone namespace {doc_hash}")
        answers = []
        for idx, question in enumerate(questions):
            try:
                print(f"[DEBUG] Querying Pinecone for question: {question}")
                query_response = co.embed(
                    texts=[question],
                    model="embed-english-v3.0",
                    input_type="search_query"
                )
                query_vec = query_response.embeddings[0]
                print(f"[DEBUG] Query vector dimension: {len(query_vec)}. Using namespace: {doc_hash}")
                results = index.query(
                    vector=query_vec, 
                    top_k=3,  # Retrieve top-3 relevant chunks
                    include_metadata=True,
                    namespace=doc_hash
                )
                if idx == 0:
                    print(f"[DEBUG] Pinecone results for first question: {results}")
                context_chunks = [match['metadata']['text'] for match in results['matches']]
                matched_sections = [match['metadata'].get('section', '') for match in results['matches']]
                answer = ask_perplexity(question, context_chunks)
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

# Route definitions under router
@router.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest, authorization: str = Header(None)):
    if authorization != f"Bearer {API_KEY}":
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

# Include router
app.include_router(router)

# Uvicorn runner for local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
