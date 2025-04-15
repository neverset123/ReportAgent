import sqlite3
# import torch
from transformers import AutoTokenizer
from pydantic import BaseModel
from typing import List
# from pydantic_sqlalchemy import sqlalchemy_to_pydantic
# from models import Paper
from training.text_model import TextClassifier
from config import config
import requests
import numpy as np

class PaperSchema(BaseModel):
    id: int
    title: str
    sentences: List[str]

# PaperSchema = sqlalchemy_to_pydantic(Paper)
def read_data_from_db(db_path, topic):
    conf = config.get_config(topic)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(f"SELECT id, title, sentences FROM articles where {conf['label']} is null").fetchall()
    papers = []
    for row in rows:
        paper = PaperSchema(
            id=row[0],
            title=row[1],
            sentences=row[2].split('\n'),
        )
        papers.append(paper)
    conn.close()
    return papers

# distilbert: used for NLP task
def split_text_into_chunks(text, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i + max_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def chunks(arr, chunk_size=2):
    for i in range(0, len(arr), chunk_size):
        yield arr[i : i + chunk_size]

def embedding_text(papers):
    api_url = "https://model-apis.semanticscholar.org/specter/v1/invoke"
    embeddings_dict = {}
    for chunk in chunks(papers):
        response = requests.post(api_url, json=chunk)
        if response.status_code != 200:
            raise RuntimeError("embedding text failed!")
        for paper in response.json()["preds"]:
            embeddings_dict[paper["paper_id"]] = paper["embedding"]
    return embeddings_dict

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

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:  # handle zero vectors case
        return 0.0
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity

def predict_relevance_emb(papers, topic):
    predictions = []
    conf = config.get_config(topic)
    paper_dict_list = []
    for paper in papers:
        sentences = " ".join(paper.sentences)
        paper_dict = {
            "paper_id": str(paper.id),
            "title": paper.title,
            "abstract": sentences,
        }
        paper_dict_list.append(paper_dict)
    embeddings_dict = embedding_text(paper_dict_list)
    for id, emb in embeddings_dict.items():
        print(id)
        count=0
        for emb_ref in conf["embedding"]:
            score = cosine_similarity(emb, emb_ref)
            print(score)
            # change to min-max strategy
            if score > 0.7:
                count += 1
        count_perc = count/len(conf["embedding"])
        print(count_perc)
        prediction = 1 if count_perc > 0.2 else 0
        predictions.append((id, prediction))
    return predictions

def write_predictions_to_db(db_path, predictions, topic):
    conf = config.get_config(topic)
    conn = sqlite3.connect(db_path)
    for id, prediction in predictions:
        conn.execute(f"UPDATE articles SET {conf['label']} = ? WHERE id = ?", (prediction, id))
        conn.commit()
    conn.close()

if __name__ == '__main__':
    db_path = 'arxiv_articles.db'
    # topics = config.get_config_names()
    topics = ["RAG", "CLIP","LLM"]
    for topic in topics:
        papers = read_data_from_db(db_path, topic)
        predictions = predict_relevance_emb(papers, topic)
        write_predictions_to_db(db_path, predictions, topic)
        print(f"Processed {len(papers)} papers for topic {topic} and wrote predictions to the database.")