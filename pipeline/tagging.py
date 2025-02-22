import sqlite3
# import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
from typing import List
# from pydantic_sqlalchemy import sqlalchemy_to_pydantic
# from models import Paper
from training.text_model import TextClassifier
from config import config

class PaperSchema(BaseModel):
    id: int
    sentences: List[str]

# PaperSchema = sqlalchemy_to_pydantic(Paper)
def read_data_from_db(db_path, topic):
    conf = config.get_config(topic)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, sentences FROM articles where {conf['label']} is null")
    rows = cursor.fetchall()
    papers = []
    for row in rows:
        paper = PaperSchema(
            id=row[0],
            sentences=row[1].split('\n'),
        )
        papers.append(paper)
    conn.close()
    return papers

def split_text_into_chunks(text, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i + max_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# predict relevance using DistilBERT
def predict_relevance(papers, topic):
    predictions = []
    conf = config.get_config(topic)
    model = TextClassifier()
    model.load_model(f"training/models/model_{conf['label']}", sklearn=False)
    for paper in papers:
        count = 0
        print(paper.id)
        sentences = " ".join(paper.sentences)
        sentence_chunks = split_text_into_chunks(sentences)
        for sentence in sentence_chunks:
            # print(sentence)
            # print(model.predict_proba([sentence]))
            if model.predict_proba([sentence])[0, 1] > 0.5:
                count += 1
        print(count/len(sentence_chunks))
        prediction = 1 if count/len(sentence_chunks) > 0.3 else 0
        predictions.append((paper.id, prediction))
    return predictions

def write_predictions_to_db(db_path, predictions, topic):
    conf = config.get_config(topic)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for id, prediction in predictions:
        cursor.execute(f"UPDATE articles SET {conf['label']} = ? WHERE id = ?", (prediction, id))
        conn.commit()
    conn.close()

if __name__ == '__main__':
    db_path = 'arxiv_articles.db'
    # topics = config.get_config_names()
    topics = ["LLM", "RAG"]
    for topic in topics:
        papers = read_data_from_db(db_path, topic)
        predictions = predict_relevance(papers, topic)
        write_predictions_to_db(db_path, predictions, topic)
        print(f"Processed {len(papers)} papers for topic {topic} and wrote predictions to the database.")