import string
from datetime import datetime

import requests
from googlesearch import search
from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from core.model import llm_ollama
from tool.tool_faq_rag import faq_rag
from tool.tool_search_web import search_using_third_party_raw
from tool.tool_stock_info import historical_lookup, company_profile, shareholder_lookup, subsidiary_lookup, \
    combined_vapfo_summary, combined_bvhl_pricemod


@tool
def weather_api(city: str | None) -> str:
    """
    Check the real-time weather in a specified city, input 'Jakarta' as default city if not provided.

    Args:
        city (str): The name of the city.

    Returns:
        str: Description of the current weather in the specified city.
    """
    print("Tool: weather called " + city)
    response = city + ", " + \
               list(search("site:weather.tomorrow.io " + city, region="id", advanced=True, num_results=1))[
                   0].description + " (data is in fahrenheit, use tool to convert to celsius)"
    print(response)
    return response


@tool
def web_query(query: str) -> str:
    """
    Search the web for real-time information, do not use this tool for Tuntun and Securities related question.

    Args:
        query (str): The question.

    Returns:
        str: Result of the search.
    """
    print("Tool: search called > " + query)
    # return search_and_get_web(query)
    return search_using_third_party_raw(query)


@tool
def fahrenheit_to_celsius(value: int) -> float:
    """
    Convert fahrenheit temperature to celsius.

    Args:
        value (int): Fahrenheit temperature.

    Returns:
        str: Celsius temperature result.
    """
    print("Tool: fahrenheit_to_celsius called")
    return (value - 32) * 5 / 9


@tool
def get_news(query: str) -> str:
    """
    Get real-time news.

    Args:
        query (str): The question.

    Returns:
        str: Result.
    """
    print("Tool: news called > " + query)
    return search_using_third_party_raw(query + " Stock Market")


@tool
def stock_price(code: str) -> str:
    """
    Check real-time stock price.

    Args:
        code (str): Stock code.

    Returns:
        str: Result of the stock price.
    """
    code = code.upper()
    print("Tool: stock_price called " + code)
    raw_res = requests.get(
        'https://ui-testing002.istock.co.id/stocks-ui/individual-stock/stock/get-stock-basic-info?accessToken=mockPass&code=' + code)
    data = raw_res.json()['data']
    res = "Latest price, stock_name: " + data['name'] + ", stock_price: " + format_rupiah(data['price']) + ", change: " + data[
        'chg'] + ", change_percent: " + data['chgPercent']
    return res


@tool
def frequently_asked(query: str) -> str:
    """
    Tuntun product, feature, registration, administration, regulation, and securities related RAG information. Use complete question as the input.

    Args:
        query (str): Stock code.

    Returns:
        str: Extract the "data" part from json as the response.
    """
    print("Tool: frequently_asked called " + query)
    return faq_rag(query)


@tool
def get_current_time() -> str:
    """
        Get current date and time.
        Args:

        Returns:
            str: Date and time

        """
    time = str(datetime.today())
    print("Tool: get_current_time " + time)
    return time


tools = [weather_api, stock_price, get_news, web_query, fahrenheit_to_celsius, frequently_asked, combined_vapfo_summary,
         combined_bvhl_pricemod, historical_lookup, company_profile, shareholder_lookup, subsidiary_lookup]
tool_node = ToolNode(tools)
llm_with_tools = llm_ollama.bind_tools(tools)

tool_mapping = {
    "get_current_time": get_current_time,
    "weather_api": weather_api,
    "stock_price": stock_price,
    "get_news": get_news,
    "web_query": web_query,
    "fahrenheit_to_celsius": fahrenheit_to_celsius,
    "frequently_asked": frequently_asked,
    "combined_vapfo_summary": combined_vapfo_summary,
    "combined_bvhl_pricemod": combined_bvhl_pricemod,
    "historical_lookup": historical_lookup,
    "company_profile": company_profile,
    "shareholder_lookup": shareholder_lookup,
    "subsidiary_lookup": subsidiary_lookup
}


def process_tools(query):
    tool_result = ""
    # tool_template = "Only use tool if question is related to the tool. Prioritize frequently_asked tool. Question: {question}"
    tool_template = "[system_date (YYYY-MM-DD):" + str(datetime.today().strftime(
        '%Y-%m-%d')) + "] Only use tool if question is related to the tool. Do not modify number formatting. Question: {question}"
    tool_prompt = PromptTemplate(
        input_variables=["question"],
        template=tool_template
    )
    final_query = tool_prompt.format(question=query)
    llm_output = llm_with_tools.invoke(final_query)
    for tool_call in llm_output.tool_calls:
        tool_output = tool_mapping[tool_call["name"].lower()].invoke(tool_call["args"])
        tool_result += str(
            "[tool_" + tool_call["name"].lower() + ": " + ToolMessage(tool_output, tool_call_id=tool_call[
                "id"]).content + "]")
    return tool_result
