# update this file to render template with jinja2 instead of BeautifulSoup
from bs4 import BeautifulSoup
from pydantic import BaseModel
from config import config
import sqlite3

class PaperSchema(BaseModel):
    id: int
    created: str
    title: str
    abstract: str
    url: str

def read_data_from_db(db_path, topic):
    conf = config.get_config(topic)
    conn = sqlite3.connect(db_path)
    # conn.row_factory = sqlite3.Row # to return dicts rather than tuples
    # The sqlite3 library in Python allows you to use conn.execute() directly, instead of creating cursor
    rows = conn.execute(f"SELECT id, created, title, abstract, url FROM articles where {conf['label']} = 1 order by created desc limit 9").fetchall()
    papers = []
    for row in rows:
        paper = PaperSchema(
            id=row[0],
            created=row[1],
            title=row[2],
            abstract=row[3],
            url=row[4],
        )
        papers.append(paper)
    conn.close()
    return papers

# Sample real data for the gallery
confs = config.get_all_configs()
gallery_data = [
    {"title": conf["name"], "summary": conf["description"], "thumb": f"images/thumbs/{str(i+1).zfill(2)}.jpg", "full": f"images/fulls/{str(i+1).zfill(2)}.jpg"} for i, conf in enumerate(confs)
]

# Nested dictionary of multiple topics data
confs = config.get_all_configs()
topics_data = {}
db_path = 'arxiv_articles.db'
for conf in confs:
    papers = read_data_from_db(db_path, conf["name"])
    topics_data[conf["name"]] = [{"icon": "fa-gem", "id": paper.id, "created": paper.created, "title": paper.title, "abstract": paper.abstract, "url": paper.url} for paper in papers]

# Read the HTML template
with open('frontend/index-template.html', 'r', encoding='utf-8') as file:
    soup = BeautifulSoup(file, 'lxml')

# Find the gallery section
gallery_section = soup.find('div', class_='gallery')

# Clear existing gallery items
gallery_section.clear()

# Fill the gallery section with real data
for item in gallery_data:
    article = soup.new_tag('article')
    
    a_tag = soup.new_tag('a', href=item['full'], **{'class': 'image'})
    img_tag = soup.new_tag('img', src=item['thumb'], alt=item['title'])
    a_tag.append(img_tag)
    
    caption_div = soup.new_tag('div', **{'class': 'caption'})
    h3_tag = soup.new_tag('h3')
    h3_tag.string = item['title']
    p_tag = soup.new_tag('p')
    p_tag.string = item['summary']
    
    actions_ul = soup.new_tag('ul', **{'class': 'actions fixed'})
    li_tag = soup.new_tag('li')
    span_tag = soup.new_tag('span', **{'class': 'button small'})
    span_tag.string = 'Details'
    li_tag.append(span_tag)
    actions_ul.append(li_tag)
    
    caption_div.append(h3_tag)
    caption_div.append(p_tag)
    caption_div.append(actions_ul)
    
    article.append(a_tag)
    article.append(caption_div)
    
    gallery_section.append(article)

# Find the section with class 'topics'
topics_section = soup.find('section', class_='topics')

# Clear the existing content in the topics section
topics_section.clear()

# Update the topics section with the nested dictionary data
for topic_title, items in topics_data.items():
    if not items:
        continue
    inner_div = soup.new_tag('div', **{'class':'inner'})
    h2_tag = soup.new_tag('h2')
    h2_tag.string = topic_title
    inner_div.append(h2_tag)
    
    items_div = soup.new_tag('div', **{'class': 'items style1 medium onscroll-fade-in'})
    for index, item in enumerate(items):
        section_tag = soup.new_tag('section')
        div_tag = soup.new_tag('div', **{'x-data': "{open: false}"})
        icon_class = 'icon style2 major ' if index == 0 else 'icon solid style2 major '
        icon_span = soup.new_tag('span', **{'class': icon_class + item['icon']})
        h3_tag = soup.new_tag('h3', **{'@click': "open = ! open", 'class': "hover:underline cursor-pointer decoration-2 decoration-green-600 text-gray-800 text-sm"})
        h3_tag.string = item['title']
        div_tag_collapse = soup.new_tag('div', **{'x-show': "open", 'x-collapse.duration.500ms': "", 'class': "text-sm text-gray-500 pt-2"})
        p_tag = soup.new_tag('p')
        p_tag.string = item['abstract']
        p_tag_url = soup.new_tag('p', **{"class": "pb-2 pt-2 text-center"})
        a_tag = soup.new_tag('a', **{"class":"underline decoration-2 text-green-600 text-md pt-2", "href":item["url"], "target":"_blank"})
        a_tag.string="full paper"
        a_tag_2 = soup.new_tag('a', **{"class":"underline decoration-2 text-green-600 text-md pt-2 ml-4", "href":f"pdf/{str(item['id'])}.pdf", "target":"_blank"})
        a_tag_2.string="report(pdf)"

        p_tag_url.append(a_tag)
        p_tag_url.append(a_tag_2)
        div_tag_collapse.append(p_tag)
        div_tag_collapse.append(p_tag_url)

        div_tag.append(icon_span)
        div_tag.append(h3_tag)
        div_tag.append(div_tag_collapse) 
        section_tag.append(div_tag)
        items_div.append(section_tag)
    
    inner_div.append(items_div)
    topics_section.append(inner_div)

# Save the newly generated HTML
with open('frontend/index.html', 'w', encoding='utf-8') as file:
    file.write(str(soup))

print("New HTML file generated and saved as 'index.html'")