import arxiv
import sqlite3
from pydantic import BaseModel
from typing import List
from config import config
from semanticscholar import SemanticScholar

class Paper(BaseModel):
    created: str
    title: str
    author: str
    abstract: str
    sentences: List[str]
    url: str
    pdf_url: str

# Function to split abstract into sentences
def split_into_sentences(text):
    import re
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    return sentence_endings.split(text)

def query_semantic_scholar(topic):
    conf = config.get_config(topic)
    query_keywords = " | ".join(conf["keywords"])
    sch = SemanticScholar()
    results = sch.search_paper(query=query_keywords, bulk=False, open_access_pdf=True, fields=['publicationDate','title','authors','abstract','url','openAccessPdf','citationCount'], sort='publicationDate:desc', limit=10)
    articles = []
    for result in results.items:
        article = Paper(
            created=result.publicationDate.strftime('%Y-%m-%d') if result.publicationDate is not None else "",
            title=result.title,
            author=",".join([author.name for author in result.authors][:3]),
            abstract=result.abstract if result.abstract is not None else "",
            sentences=split_into_sentences(result.abstract) if isinstance(result.abstract, str) else [],
            url=result.url,
            pdf_url=result.openAccessPdf['url']
        )
        articles.append(article)
    return articles

# Function to query arXiv API for a specific topic
def query_arxiv(topic):
    conf = config.get_config(topic)
    query_keywords = " AND abs:".join(conf["keywords"])
    query = f"abs:{query_keywords} AND (cat:cs.CV OR cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:cs.NE OR cat:cs.RO)"
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    articles = []
    for result in search.results():
        article = Paper(
            created=result.published.strftime('%Y-%m-%d'),
            title=result.title,
            author=",".join([author.name for author in result.authors][:3]),
            abstract=result.summary,
            sentences=split_into_sentences(result.summary),
            url=result.entry_id,
            pdf_url=result.pdf_url
        )
        articles.append(article)
    return articles

# Function to save articles to SQLite database
def save_to_db(articles, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            created TEXT,
            title TEXT,
            author TEXT,
            abstract TEXT,
            sentences TEXT,
            url TEXT UNIQUE,
            pdf_url TEXT UNIQUE
        )
    ''')
    # cursor.execute("update articles set llm = NULL where id >= 0")
    # cursor.execute("drop table articles")
    for article in articles:
        try:
            cursor.execute('''
                INSERT INTO articles (created, title, author, abstract, sentences, url, pdf_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (article.created, article.title, article.author, article.abstract, '\n'.join(article.sentences), article.url, article.pdf_url))
        except sqlite3.IntegrityError:
            print(f"Article with URL {article.url} already exists in the database.")
    conn.commit()
    conn.close()

# Main function to query and save articles
def main(topics, db_path):
    for topic in topics:
        articles = query_arxiv(topic)
        # articles = query_semantic_scholar(topic)
        save_to_db(articles, db_path)
        print(f"Processed {len(articles)} articles for topic {topic}.")

# Example usage
if __name__ == '__main__':
    # topics = config.get_config_names()
    topics = ["LLM", "RAG"]
    db_path = 'arxiv_articles.db'
    main(topics, db_path)