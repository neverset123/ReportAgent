import sqlite3
from pydantic import BaseModel
import sqlite3

class SlideSchema(BaseModel):
    id: int
    title: str
    author: str
    date: str
    problem: str
    approach: str
    model: str
    dataset: str
    evaluation: str
    conclusion: str

def get_report_data_from_db(db_path):
    print("Read data from db...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("select a.id, a.title, a.author, a.created, r.problem, r.approach, r.model, r.dataset, r.evaluation, r.conclusion from articles a inner join report r using (id) where a.id >19 order by a.created")
    rows = cursor.fetchall()
    data = []
    for row in rows:
        ele = SlideSchema(
            id=row[0],
            title=row[1],
            author=row[2],
            date=row[3],
            problem=row[4],
            approach=row[5],
            model=row[6],
            dataset=row[7],
            evaluation=row[8],
            conclusion=row[9]
        )
        data.append(ele)
    conn.close()
    return data

def read_markdown_template(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def generate_markdown(template, data):
    for ele in data:
        content = template.format(
            title=ele.title,
            author=ele.author,
            date=ele.date,
            problem=ele.problem,
            approach=ele.approach,
            model=ele.model,
            dataset=ele.dataset,
            evaluation=ele.evaluation,
            conclusion=ele.conclusion
        )
        filepath = f"./slides/md/{ele.id}.md"
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
    print("Markdown content generated and saved successfully.")

if __name__ == "__main__":
    template_path = "slides/slides_template.md"
    db_path = 'arxiv_articles.db'
    markdown_template = read_markdown_template(template_path)
    report_data = get_report_data_from_db(db_path)
    generate_markdown(markdown_template, report_data)

