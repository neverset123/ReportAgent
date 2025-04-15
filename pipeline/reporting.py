# extract information from arxiv paper with multi-agent framework swarm (Requires Python 3.10+)
import requests
import os
import openai
import sqlite3
import PIL
from config import config
from pydantic import BaseModel
from dotenv import load_dotenv
from swarm import Agent, Swarm
from arxiv2text import arxiv_to_text
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
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
layout: two-cols
---

# Key Approach
{approach}
::right::
<img border="rounded" src="{img1}" alt="">
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
<img border="rounded" src="{img2}" alt="">
---
layout: two-cols
---

# Evaluation 
{evaluation}
::right::
<img border="rounded" src="{img3}" alt="">
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


config_parser = ConfigParser({"output_format": "markdown"})
converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(),
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service()
)

def extract_page_figure(s):
    parts = s.split('_')
    page_number = int(parts[2])
    figure_number = int(parts[4].replace('.jpeg', ''))
    return (page_number, figure_number)

def extract_image(elements):
    img_path_list = []
    new_width = 577
    for ele in elements:
        img_paths = []
        os.makedirs("data/", exist_ok=True)
        os.makedirs(f"slides/md/{str(ele.id)}", exist_ok=True)
        response = requests.get(ele.url)
        file_path = f"data/arxiv_{ele.id}.pdf"
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"PDF downloaded successfully")
        else:
            print(f"Failed to download: {response.status_code}")

        rendered = converter(file_path)
        _, _, images = text_from_rendered(rendered)
        for key, value in images.items():
            img_path = f"slides/md/{str(ele.id)}/{key}"
            original_width, original_height = value.size
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
            resized_img = value.resize((new_width, new_height), PIL.Image.LANCZOS)
            resized_img.save(img_path)
            img_paths.append(f"/{str(ele.id)}/{key}")
        img_path_list.append(sorted(img_paths, key=extract_page_figure))
    return img_path_list

def proprocess_paper(context_variables):
    paper_text = arxiv_to_text(context_variables["url"])
    return f"Summary: {paper_text[:4000]}"

def preprocess_template(context_variable):
    template_updated = (response_template.replace("{title}", context_variable["title"])
                             .replace("{author}",  context_variable["author"])
                             .replace("{date}", context_variable["date"]))
    img_list = []
    if len(context_variable["img"])>=3:
        img_list = [context_variable["img"][0], context_variable["img"][1], context_variable["img"][2]]
    elif len(context_variable["img"])==2:
        img_list = [context_variable["img"][0], context_variable["img"][1], ""]
    elif len(context_variable["img"])==1:
        img_list = [context_variable["img"][0], "", ""]
    else:
        img_list = ["", "", ""]

    new_template = (template_updated.replace("{img1}", img_list[0])
                            .replace("{img2}", img_list[1])
                            .replace("{img3}", img_list[2])
                    )
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

def get_urls_from_db(db_path, topic):
    conf = config.get_config(topic)
    conn = sqlite3.connect(db_path)
    rows =  conn.execute(f"select id, title, author, created, pdf_url from articles where report is null and {conf['label']} = 1").fetchall()
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

def generate_md(elements, db_path, img_path_list):
    conn = sqlite3.connect(db_path)
    for index,  ele in enumerate(elements):
        response = client.run(
            agent=Paper_agent,
            messages=[{"role": "user", "content": user_message}],
            context_variables={"url": ele.url, "title": ele.title, "author": ele.author, "date": ele.date, "img": img_path_list[index]}
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
        conn.execute(f"UPDATE articles SET report = ? WHERE id = ?", (True, ele.id))
        conn.commit()
    conn.close()

if __name__ == "__main__":
    topics = ["RAG", "CLIP","LLM"]
    db_path = 'arxiv_articles.db'
    for topic in topics:
        data = get_urls_from_db(db_path, topic)
        img_path_list = extract_image(data)
        generate_md(data, db_path, img_path_list)
