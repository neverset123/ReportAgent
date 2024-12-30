from flask import Flask, request, jsonify
from langchain import LLMMathChain, SerpAPIWrapper, SQLDatabase
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import AzureChatOpenAI
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

llm = AzureChatOpenAI(model=os.getenv("CHAT_MODEL"), 
                        azure_endpoint=os.getenv("ENDPOINT"),
                        azure_deployment=os.getenv("CHAT_MODEL"),
                        openai_api_version=os.getenv("API_VERSION"),
                        api_key=os.getenv("APIKEY")
)
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
db = SQLDatabase.from_uri("sqlite:///arxiv_articles.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_chain = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

tools = [
    Tool(
        name="SearchTool",
        func=search.run,
        description="search and answer"
    ),
    Tool(
        name="MathTool",
        func=llm_math_chain.run,
        description="mathmatic calculation"
    ),
    Tool(
        name="QueryTool",
        func=db_chain.run,
        description="query db"
    )
]

agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

@app.route('/api/chat', methods=['POST'])
def chat():
    user_question = request.json.get('message')
    if not user_question:
        return jsonify({'error': 'No question provided'}), 400

    response = agent.run(user_question)

    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(debug=True)