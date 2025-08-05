# db.py
# PostgreSQL/SQLAlchemy async setup for logging queries and answers
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:1234@localhost:5432/hackrx")

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class QueryLog(Base):
    __tablename__ = "query_logs"
    id = Column(Integer, primary_key=True, index=True)
    document_url = Column(String, index=True)
    question = Column(Text)
    answer = Column(Text)
    matched_text = Column(Text)
    rationale = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
