import requests
from mcp.server.fastmcp import FastMCP
from typing import List

# Initialize the MCP server
mcp = FastMCP("NewsServer")

# Define your news_summary tool (slightly adapted for MCP)
@mcp.tool()
def news_summary(code: str) -> str:
    """
    Get the latest 5 news of related stock code.

    Args:
        code (str): Stock code (e.g., "AAPL").

    Returns:
        str: A string representation of the 5 latest news items related to the stock code.
    """
    code = code.upper()
    raw_res = requests.post(
        "http://10.192.2.42:8080/news/latest-news",
        headers={'Content-Type': 'application/json'},
        json={"secCodes": [code]},
        timeout=300, allow_redirects=True
    )
    r = raw_res.json()
    if "message" in r and r["message"] != "Success":
        return "No news found."

    result = []
    if "data" in r and "list" in r["data"]:
        items = r["data"]["list"]
        for item in items:
            y = item["published_date"].split("T")
            news_date = y[0]
            news_time = y[1]
            news_title = item["title"]
            news_summary_text = item["summarize_llm"]
            x = {
                "stock_code": code,
                "news_date": news_date,
                "news_time": news_time,
                "news_title": news_title,
                "news_summary": news_summary_text
            }
            result.append(x)
    result_str = "\n".join([str(item) for item in result])
    print(f"Tool: news_summary code: {code}. total news fetched: {len(result)}")
    return result_str if result else "No news found."

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")
