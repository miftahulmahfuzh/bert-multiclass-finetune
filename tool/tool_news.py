import requests
from langchain_core.tools import tool
from typing import List

@tool
def news_summary(code: str):
    """
    Get the latest 5 news of related stock code.

    Args:
        code (str): Stock code.

    Returns:
        list: 5 latest news related to stock code.
    """
    code = code.upper()
    raw_res = requests.post(
        "http://10.192.2.42:8080/news/latest-news",
        headers={'Content-Type': 'application/json'},
        json={"secCodes": [code]},
        timeout=300, allow_redirects=True
    )
    r = raw_res.json()
    if "message" in r:
        message = r["message"]
        if message != "Success":
            return []

    result = []
    result_str = ""
    if "data" in r:
        data = raw_res.json()["data"]
        if "list" in data:
            items = data["list"]
            for item in items:
                y = item["published_date"].split("T")
                news_date = y[0]
                news_time = y[1]
                news_title = item["title"]
                news_summary = item["summarize_llm"]
                x = {
                    "stock_code": code,
                    "news_date": news_date,
                    "news_time": news_time,
                    "news_title": news_title,
                    "news_summary": news_summary
                }
                result_str += str(x)
                result.append(x)
    print(f"Tool: news_summary code: {code}. total news fetched: {len(result)}")
    return result_str
