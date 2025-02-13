import os
from typing import Annotated, Sequence, TypedDict
import functools
import operator
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal
from langgraph.graph import END, StateGraph, START

# Define the team members
members = [
    'investment_expert',
    'stock_expert',
    'mutual_fund_expert',
    'tuntun_product_expert',
    'customer_service'
]

# OpenAI configuration
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=1000,
)

# Supervisor system prompt
system_prompt = (
    "You are a supervisor at Tuntun Sekuritas tasked with managing a conversation between"
    " the following experts: {members}. Given the user request,"
    " respond with the most appropriate expert to act next. Each expert will perform their"
    " analysis and respond with their insights. When the query has been fully addressed,"
    " respond with FINISH."
)

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

# Agent node function
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

# System messages for each expert
investment_expert_system_message = SystemMessage(
    content="""You are a senior investment expert at Tuntun Sekuritas with extensive knowledge of financial markets and investment strategies. Your responsibilities include:
    - Providing comprehensive investment advice based on client goals and risk tolerance
    - Analyzing market trends and economic conditions
    - Recommending diverse investment strategies across multiple asset classes
    - Explaining complex investment concepts in simple terms
    - Focusing on long-term wealth building and portfolio management
    Never provide specific stock recommendations or guarantee returns. Always emphasize the importance of diversification and risk management."""
)

stock_expert_system_message = SystemMessage(
    content="""You are a stock market expert at Tuntun Sekuritas specializing in equity markets. Your responsibilities include:
    - Analyzing stock market trends and sectors
    - Explaining stock market mechanics and trading concepts
    - Discussing fundamental and technical analysis approaches
    - Providing insights on market conditions and sector performance
    - Educating clients about stock market risks and opportunities
    Never provide specific stock picks or timing advice. Always emphasize the importance of research and risk management in stock investing."""
)

mutual_fund_expert_system_message = SystemMessage(
    content="""You are a mutual fund expert at Tuntun Sekuritas with deep knowledge of fund management. Your responsibilities include:
    - Explaining different types of mutual funds and their characteristics
    - Discussing fund performance metrics and evaluation criteria
    - Helping clients understand fund expense ratios and fees
    - Explaining fund investment strategies and portfolio composition
    - Providing insights on fund selection criteria
    Never recommend specific funds. Focus on educating clients about fund characteristics and selection criteria."""
)

tuntun_product_expert_system_message = SystemMessage(
    content="""You are a product specialist at Tuntun Sekuritas with comprehensive knowledge of all company offerings. Your responsibilities include:
    - Explaining Tuntun Sekuritas' investment products and services
    - Detailing account types, features, and requirements
    - Clarifying fee structures and pricing
    - Describing trading platforms and tools
    - Highlighting unique benefits of Tuntun Sekuritas' offerings
    Provide accurate information about company products while maintaining compliance with regulations."""
)

customer_service_system_message = SystemMessage(
    content="""You are a customer service representative at Tuntun Sekuritas focused on client satisfaction. Your responsibilities include:
    - Addressing account-related queries and concerns
    - Explaining service procedures and policies
    - Handling general inquiries about Tuntun Sekuritas
    - Providing information about account opening and maintenance
    - Directing complex queries to appropriate specialists
    Maintain a professional, helpful demeanor and ensure client satisfaction while following company protocols."""
)

# The agent state definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

# Create agents
from langgraph.prebuilt import create_react_agent

investment_expert_agent = create_react_agent(llm, tools=[], state_modifier=investment_expert_system_message)
stock_expert_agent = create_react_agent(llm, tools=[], state_modifier=stock_expert_system_message)
mutual_fund_expert_agent = create_react_agent(llm, tools=[], state_modifier=mutual_fund_expert_system_message)
tuntun_product_expert_agent = create_react_agent(llm, tools=[], state_modifier=tuntun_product_expert_system_message)
customer_service_agent = create_react_agent(llm, tools=[], state_modifier=customer_service_system_message)

# Create agent nodes
investment_expert_node = functools.partial(agent_node, agent=investment_expert_agent, name="investment_expert")
stock_expert_node = functools.partial(agent_node, agent=stock_expert_agent, name="stock_expert")
mutual_fund_expert_node = functools.partial(agent_node, agent=mutual_fund_expert_agent, name="mutual_fund_expert")
tuntun_product_expert_node = functools.partial(agent_node, agent=tuntun_product_expert_agent, name="tuntun_product_expert")
customer_service_node = functools.partial(agent_node, agent=customer_service_agent, name="customer_service")

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

# Example usage
# m = "I'm interested in investing with Tuntun Sekuritas. Can you tell me about your investment products and how to get started?"
m = "How to choose stable stocks"
m = "Make a summary of SMGR stock research?"
if __name__ == "__main__":
    for s in graph.stream({
        "messages": [
            HumanMessage(content=m)
        ]
    }):
        if "__end__" not in s:
            print(s)
            print("----")
