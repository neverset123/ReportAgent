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

response_template = """
---
theme: seriph
background: https://cover.sli.dev
title: Report Cover
info: |
  ## Slidev Template
  Presentation slides for papers.
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
export:
  format: pdf
  timeout: 600000 
---

## {title}
- {author}
- {date}
---
transition: fade-out
---

# Table of Contents
<Toc text-sm minDepth="1" maxDepth="2" />
---
transition: slide-right
---

# Problem Statement
{problem}
---

# Key Approach
{approach}
---
transition: slide-up
level: 2
---

# Key Steps/Models
{model}
---
transition: slide-up
level: 2
---

# Dataset 
{dataset}
---

# Evaluation 
{evaluation}
---

# Conclusion
{conclusion}
---
class: px-20
---
"""

user_message = ("Instructions: " + 
                "- Problem Statement: Identify what specific problem they are solving, outlines the primary challenge, gap, or issue it aims to address. "
                "- Key Approach: Summarize the main approach proposed by the authors, including any novel techniques, algorithms, or frameworks introduced. " 
                "- Key Steps/Models: Describe the key steps, architecture, modules, or stages involved, and explain how each contributes to the overall method. " 
                "- Dataset Details: Provide an overview of the datasets and benchmarks used, include dataset size, type, source and public availability. " 
                "- Evaluation Methods and Metrics: Detail the evaluation process to assess performance, include the methods, benchmarks, and metrics employed. " 
                "- Conclusion: Summarize the conclusions, include the significance of the findings, potential applications, limitations and future work.\n " 
                "Ensure that the summary is clear and concise, all details are accurate and faithfully represent the content of the original paper.")

def proprocess_paper(context_variables):
    paper_text = arxiv_to_text(context_variables["url"])
    return f"Summary: {paper_text[:4000]}"

def preprocess_template(context_variable):
    new_template = (response_template.replace("{title}", context_variable["title"])
                             .replace("{author}",  context_variable["author"])
                             .replace("{date}", context_variable["date"]))
    instruction= ("You are an expert in summarizing scientific papers. " + \
    "Goal is to create concise and informative summaries, with each section around 100-200 words. " + \
    f"Structure the output into the string {new_template} without any prefix or postfix")
    return instruction


azure_openai_client = openai.AzureOpenAI(
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("APIKEY")
)
client= Swarm(client=azure_openai_client)
Paper_agent = Agent(
    name="Paper Agent",
    model=os.getenv("CHAT_MODEL"),
    instructions=preprocess_template,
    functions=[proprocess_paper]
)

class ReportSchema(BaseModel):
    id: int
    title: str
    author: str
    date: str
    url: str

def get_urls_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # cursor.execute("select id, pdf_url from articles where id not in (select id from report)")
    cursor.execute("select id, title, author, created, pdf_url from articles where report is null")
    rows = cursor.fetchall()
    data = []
    for row in rows:
        ele = ReportSchema(
            id=row[0],
            title=row[1],
            author=row[2],
            date=row[3],
            url=row[4]
        )
        data.append(ele)
    conn.close()
    return data

def generate_md(elements, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for ele in elements:
        response = client.run(
            agent=Paper_agent,
            messages=[{"role": "user", "content": user_message}],
            context_variables={"url": ele.url, "title": ele.title, "author": ele.author, "date": ele.date}
        )
        try:
            res_md = response.messages[-1]["content"]
        except Exception as e:
            print(f"Unexpected error for paper {ele.id}: {e}")
            res_md = ""
        filepath = f"./slides/md/{ele.id}.md"
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(res_md)
        print(f"Markdown content for article {ele.id} is saved successfully.")
        cursor.execute(f"UPDATE articles SET report = ? WHERE id = ?", (True, ele.id))
        conn.commit()
    conn.close()

if __name__ == "__main__":
    db_path = 'arxiv_articles.db'
    data = get_urls_from_db(db_path)
    metadatas = generate_md(data, db_path)
