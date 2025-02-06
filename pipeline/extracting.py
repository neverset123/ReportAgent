# extract information from arxiv paper with multi-agent framework swarm (Requires Python 3.10+)
import json
import os
import openai
import sqlite3
from pydantic import BaseModel
from dotenv import load_dotenv
from swarm import Agent, Swarm
from arxiv2text import arxiv_to_text
load_dotenv()

response_template = {
    "problem": "",
    "approach": "",
    "model": "",
    "dataset": "",
    "evaluation": "",
    "conclusion": ""
}

user_message = ("Instructions:" + 
                "- Problem Statement: Identify the motivation for the study and what specific problem they are solving. " 
                "It typically outlines the primary challenge, gap, or issue the research aims to address. "
                "- Key Approach: Summarize the main approach or model proposed by the authors. " 
                "Focus on the core idea behind their method, including any novel techniques, algorithms, or frameworks introduced. " 
                "- Key Steps/Models: Identify and describe the key steps in the model. " 
                "Break down the architecture, modules, or stages involved, and explain how each contributes to the overall method. " 
                "Explain how the authors trained or finetuned their model." 
                "Include details on the training process, loss functions, optimization techniques, " 
                "and any specific strategies used to improve the model's performance." 
                "- Dataset Details: Provide an overview of the datasets used in the study." 
                "Include information on the size, type and source. Mention whether the dataset is publicly available " 
                "and if there are any benchmarks associated with it." 
                "- Evaluation Methods and Metrics: Detail the evaluation process used to assess the model's performance. " 
                "Include the methods, benchmarks, and metrics employed." 
                "- Conclusion: Summarize the conclusions drawn by the authors. Include the significance of the findings, " 
                "any potential applications, limitations acknowledged by the authors, and suggested future work.\n " 
                "Ensure that the summary is clear and concise, avoiding unnecessary jargon or overly technical language." 
                "Aim to be understandable to someone with a general background in the field." 
               " Ensure that all details are accurate and faithfully represent the content of the original paper.")

def proprocess_paper(context_variables):
    paper_text = arxiv_to_text(context_variables["url"])
    return f"Summary: {paper_text[:4000]}"

azure_openai_client = openai.AzureOpenAI(
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("APIKEY")
)
client= Swarm(client=azure_openai_client)
Paper_agent = Agent(
    name="Paper Agent",
    model=os.getenv("CHAT_MODEL"),
    instructions="You are an expert in summarizing scientific papers. " + \
                "Goal is to create concise and informative summaries, with each section around 100-200 words, " + \
                "focusing on the problem statement, core approach, model training methodology, dataset details, " + \
                "evaluation, and conclusions presented in the paper. " + \
                f"output each part as formated text for slidev and fill into the string {json.dumps(response_template)} without any prefix or postfix",
    functions=[proprocess_paper]
)

class UrlSchema(BaseModel):
    id: int
    url: str

def get_urls_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("select id, pdf_url from articles where id not in (select id from report)")
    rows = cursor.fetchall()
    urls = []
    for row in rows:
        ele = UrlSchema(
            id=row[0],
            url=row[1],
        )
        urls.append(ele)
    conn.close()
    return urls

def extract_metadata(urls):
    metadatas = []
    for ele in urls:
        response = client.run(
            agent=Paper_agent,
            messages=[{"role": "user", "content": user_message}],
            context_variables={"url": ele.url}
        )
        try:
            res_dict = json.loads(response.messages[-1]["content"])
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError for paper {ele.id}: {e}")
            res_dict = {}
        except TypeError as e:
            print(f"TypeError for paper {ele.id}: {e}")
            res_dict = {}
        except Exception as e:
            print(f"Unexpected error for paper {ele.id}: {e}")
            res_dict = {}
        metadatas.append((ele.id, res_dict))
        print(f"metadta of paper {ele.id} is extracted!")
    return metadatas

# Function to write predictions back to SQLite database
def save_metadata_to_db(db_path, metadatas):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS report (
            id INTEGER PRIMARY KEY,
            problem TEXT,
            approach TEXT,
            model TEXT,
            dataset TEXT,
            evaluation TEXT,
            conclusion TEXT
        )
    ''')

    for id, metadata in metadatas:
        if not metadata:
            continue
        try:
            cursor.execute('''
                INSERT INTO report (id, problem, approach, model, dataset, evaluation, conclusion)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (id, metadata["problem"], metadata["approach"], metadata["model"], metadata["dataset"], metadata["evaluation"], metadata["conclusion"]))
        except sqlite3.IntegrityError:
            print(f"report with id {id} already exists in the database.")
    conn.commit()
    conn.close()
    print("all metadata are saved in db!")

if __name__ == "__main__":
    db_path = 'arxiv_articles.db'
    urls = get_urls_from_db(db_path)
    metadatas = extract_metadata(urls)
    save_metadata_to_db(db_path, metadatas)

