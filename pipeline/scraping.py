import arxiv
import sqlite3
from pydantic import BaseModel
from typing import List
from config import config

class Paper(BaseModel):
    created: str
    title: str
    abstract: str
    sentences: List[str]
    url: str

# Function to split abstract into sentences
def split_into_sentences(text):
    import re
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    return sentence_endings.split(text)

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
            abstract=result.summary,
            sentences=split_into_sentences(result.summary),
            url=result.entry_id
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
            abstract TEXT,
            sentences TEXT,
            url TEXT UNIQUE
        )
    ''')
    # cursor.execute("update articles set llm = NULL where id >= 0")
    # cursor.execute("drop table articles")
    for article in articles:
        try:
            cursor.execute('''
                INSERT INTO articles (created, title, abstract, sentences, url)
                VALUES (?, ?, ?, ?, ?)
            ''', (article.created, article.title, article.abstract, '\n'.join(article.sentences), article.url))
        except sqlite3.IntegrityError:
            print(f"Article with URL {article.url} already exists in the database.")
    conn.commit()
    conn.close()

# Main function to query and save articles
def main(topics, db_path):
    for topic in topics:
        articles = query_arxiv(topic)
        save_to_db(articles, db_path)
        print(f"Processed {len(articles)} articles for topic {topic}.")

# Example usage
if __name__ == '__main__':
    topics = config.get_config_names()
    db_path = 'arxiv_articles.db'
    main(topics, db_path)