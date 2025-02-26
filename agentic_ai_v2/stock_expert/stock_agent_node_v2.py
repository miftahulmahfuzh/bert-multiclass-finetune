from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict
from stock_expert.stock_query_preprocessing_v2 import (
    process_stock_query,
    analyze_stock_data,
)
from search_tool import (
    search_and_summarize,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal
from langchain_openai import ChatOpenAI

# Define the LLM with the same configuration as in main.py
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=1000,
)

# Define a structured output model for the category
class CategoryResponse(BaseModel):
    category: Literal["category_1", "category_2"]

# Define the categorization prompt
categorization_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Your task is to categorize the user's query into one of two categories:

**Category 1**: The user is asking for an analysis of a specific stock within a specific timeframe. The query must mention both a particular stock (e.g., BBCA, BAYU) and a specific time period (e.g., last 7 days, last 14 weeks, from the first week of January this year). Examples include:
Examples of Category 1:
- 'Give me an analysis on BBCA on the last 7 days.'
- 'Analyze BAYU stock over the past 14 weeks.'
- 'How has BBCA performed from January to March this year?'
- 'What is the current price of BBCA?'

**Category 2**: Any other type of query that does not fit into Category 1. This includes queries that:
- Ask about stocks but do not specify a timeframe.
- Ask about a timeframe but do not specify a particular stock.
- Are unrelated to stock analysis.
Examples of Category 2:
- What bullish signal is most suitable for PTBA shares?
- How has the stock market performed in the last month?
- Tell me about mutual funds.
- Analyze BBCA stock.

Please respond with only 'category_1' if the query matches Category 1, or 'category_2' if it matches Category 2."""),
    ("human", "{query}")
])

# Create the categorization chain
categorization_chain = categorization_prompt | llm.with_structured_output(CategoryResponse)

def stock_expert_agent_node(state, agent, name):
    """
    Modified agent node function specifically for stock expert to process stock queries
    using stock analysis tools instead of web search.
    """
    # Get the last message content
    last_message = state["messages"][-1].content

    # Categorize the query using the LLM
    category_response = categorization_chain.invoke({"query": last_message})
    category = category_response.category

    result = {}
    if category == "category_1":
        print(f"IN CATEGORY 1")
        # Process the stock query to get date and stock information
        query_info = process_stock_query(last_message)
        if "error" in query_info:
            error_response = f"Error processing query: {query_info['error']}"
            return {"messages": [HumanMessage(content=error_response, name=name)]}
        # Get the stock analysis
        analysis_results = analyze_stock_data(last_message)
        # Create context with the analysis results
        context = f"\nStock Analysis Results:\n{analysis_results['content']}"
        # Add analysis results to agent's input
        modified_state = {
            "messages": state["messages"] + [
                SystemMessage(content=f"Here is the stock analysis:\n{context}")
            ]
        }
        # Get agent's response
        result = agent.invoke(modified_state)
        # Prepare the final response with the analysis
        final_response = (
            f"{result['messages'][-1].content}\n\n"
            f"Analysis period: {query_info['start_date']} to {query_info['end_date']}\n"
            f"Stock Code: {query_info['stock_code']}"
        )
        result = {"messages": [HumanMessage(content=final_response, name=name)]}
    elif category == "category_2":
        print(f"IN CATEGORY 2")
        # Perform web search
        search_results = search_and_summarize(last_message)
        # Add search results to the context
        context = (
            f"\nWeb Search Results:\n{search_results['content']}\n\n"
            f"References:\n" + "\n".join([f"- {url}" for url in search_results['references']])
        )
        # Add search results to agent's input
        modified_state = {
            "messages": state["messages"] + [
                SystemMessage(content=f"Here are relevant web search results:\n{context}")
            ]
        }
        # Get agent's response
        result = agent.invoke(modified_state)
        # Include references in the response
        final_response = (
            f"{result['messages'][-1].content}\n\nReferences used:\n" +
            "\n".join([f"- {url}" for url in search_results['references']])
        )
        result = {"messages": [HumanMessage(content=final_response, name=name)]}
    return result
