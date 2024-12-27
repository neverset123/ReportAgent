from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Paper(Base):
    __tablename__ = 'articles'
    id = Column(Integer, primary_key=True)
    created = Column(String)
    title = Column(String)
    abstract = Column(String)
    sentences = Column(Text)
    url = Column(String)
    rag = Column(Boolean)
    llm = Column(Boolean)
    ad = Column(Boolean)
    mining = Column(Boolean)
    clip = Column(Boolean)

DATABASE_URL = "sqlite:///arxiv_articles.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)