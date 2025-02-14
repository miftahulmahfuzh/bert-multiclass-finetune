from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from typing import Dict, List

# Agent node function
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

# Custom web search tools
@tool
def search_and_summarize(query: str) -> Dict:
    """Search the web and return results with references"""
    tavily_search = TavilySearchResults(max_results=5)
    results = tavily_search.invoke(query)
    # print(f"TAVILY RAW RESULTS: {results}")

    ss_result = {"content": "No References", "references": []}

    if isinstance(results, list):

        # Extract URLs and content
        references = [result["url"] for result in results]
        content = "\n\n".join([f"- {result['content']}" for result in results])

        return {
            "content": content,
            "references": references
        }
    return ss_result

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Scrape content from provided URLs"""
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return "\n\n".join([
            f'Source: {doc.metadata.get("source", "")}\n{doc.page_content}'
            for doc in docs
        ])
    except Exception as e:
        return f"Error scraping webpages: {str(e)}"

# Modified agent node function to include web search
def agent_node(state, agent, name):
    # Get the last message content
    last_message = state["messages"][-1].content

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

    return {"messages": [HumanMessage(content=final_response, name=name)]}
