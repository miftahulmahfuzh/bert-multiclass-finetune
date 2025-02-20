import environments
import json
import functools
import operator
from pydantic import BaseModel
from typing import Literal, Annotated, Sequence, TypedDict, List, Dict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langchain_community.document_loaders import WebBaseLoader

# Define the team members
members = [
    'investment_expert',
    'stock_expert',
    'mutual_fund_expert',
    'tuntun_product_expert',
    'customer_service'
]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=1000,
)

from system_messages import system_prompt

# Define the possible routing options
options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal[*options]

# Supervisor prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Given the conversation above, which expert should act next?"
        " Or should we FINISH? Select one of: {options}",
    ),
]).partial(options=str(options), members=", ".join(members))

# Supervisor agent function
def supervisor_agent(state):
    supervisor_chain = prompt | llm.with_structured_output(routeResponse)
    return supervisor_chain.invoke(state)

from system_messages import (
    investment_expert_system_message,
    stock_expert_system_message,
    mutual_fund_expert_system_message,
    tuntun_product_expert_system_message,
    customer_service_system_message
)

from search_tool import (
    TavilySearchResults,
    search_and_summarize,
    scrape_webpages,
    agent_node
)
from stock_expert.stock_query_preprocessing import (
    process_stock_query,
    analyze_stock_data,
)
from stock_expert.stock_agent_node import (
    stock_expert_agent_node
)

# Create agents with tools
from langgraph.prebuilt import create_react_agent

tools = [search_and_summarize, scrape_webpages]
stock_tools = [process_stock_query, analyze_stock_data]

investment_expert_agent = create_react_agent(llm, tools=tools, state_modifier=investment_expert_system_message)
stock_expert_agent = create_react_agent(llm, tools=stock_tools, state_modifier=stock_expert_system_message)
mutual_fund_expert_agent = create_react_agent(llm, tools=tools, state_modifier=mutual_fund_expert_system_message)
tuntun_product_expert_agent = create_react_agent(llm, tools=tools, state_modifier=tuntun_product_expert_system_message)
customer_service_agent = create_react_agent(llm, tools=tools, state_modifier=customer_service_system_message)

# Create agent nodes with web search capability
investment_expert_node = functools.partial(agent_node, agent=investment_expert_agent, name="investment_expert")
stock_expert_node = functools.partial(stock_expert_agent_node, agent=stock_expert_agent, name="stock_expert")
mutual_fund_expert_node = functools.partial(agent_node, agent=mutual_fund_expert_agent, name="mutual_fund_expert")
tuntun_product_expert_node = functools.partial(agent_node, agent=tuntun_product_expert_agent, name="tuntun_product_expert")
customer_service_node = functools.partial(agent_node, agent=customer_service_agent, name="customer_service")

# The agent state definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("investment_expert", investment_expert_node)
workflow.add_node("stock_expert", stock_expert_node)
workflow.add_node("mutual_fund_expert", mutual_fund_expert_node)
workflow.add_node("tuntun_product_expert", tuntun_product_expert_node)
workflow.add_node("customer_service", customer_service_node)
workflow.add_node("supervisor", supervisor_agent)

# Add edges from each expert back to supervisor
for member in members:
    workflow.add_edge(member, "supervisor")

# Add conditional edges from supervisor to experts
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

# Add entry point
workflow.add_edge(START, "supervisor")

# Compile the graph
graph = workflow.compile()

from queries import m

def save_to_file(user_query, chatbot_answer):
    with open("cache/input_output.txt", "a") as f:
        f.write(f"input:\n{user_query}\n\noutput:\n{chatbot_answer}\n\n{'-'*40}\n")

chatbot_messages = []
if __name__ == "__main__":
    config = {"recursion_limit": 100}
    for s in graph.stream({
        "messages": [
            HumanMessage(content=m)
        ]
    }, config=config):
        if "__end__" not in s:
            print(s)
            for member in members:
                if member in s:
                    k = s[member]["messages"][0].content
                    # print(f"K KEYS: {k.content}")
                    chatbot_messages.append(k)
            print("----")
    save_to_file(m, chatbot_messages[-1])
