from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict

from stock_expert.stock_query_preprocessing import (
    process_stock_query,
    analyze_stock_data,
)

def stock_expert_agent_node(state, agent, name):
    """
    Modified agent node function specifically for stock expert to process stock queries
    using stock analysis tools instead of web search.
    """
    # Get the last message content
    last_message = state["messages"][-1].content

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

    return {"messages": [HumanMessage(content=final_response, name=name)]}
