import os

from huggingface_hub import login
from langchain_community.tools.tavily_search import TavilySearchResults
from transformers import HfEngine, ReactCodeAgent

token = ""
login(token)
os.environ["TAVILY_API_KEY"] = ""
model_id = "meta-llama/Llama-3.2-3B-Instruct"
llm_engine = HfEngine(model_id)
# Initialize the agent with both tools
agent = ReactCodeAgent(tools=[TavilySearchResults()], llm_engine=llm_engine)
#
response = agent.run("search the news regarding trump shooting")
print(response)
