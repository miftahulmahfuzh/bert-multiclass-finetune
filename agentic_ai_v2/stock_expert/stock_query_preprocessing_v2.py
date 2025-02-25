from datetime import datetime
from langchain_core.tools import tool
from typing import Dict
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import xml.etree.ElementTree as ET

@tool
def process_stock_query(query: str) -> Dict:
    """
    Process a natural language query about stock data to extract stock code and date range.
    Uses LLM to interpret semantic date expressions in natural language.
    """
    # Extract stock code - looking for capital letters possibly followed by numbers
    stock_match = re.search(r'([A-Z]+\d*)', query)
    if not stock_match:
        return {"error": "No valid stock code found in query"}
    stock_code = stock_match.group(1)

    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=100,
    )

    # Create prompt template for date range extraction
    date_prompt = ChatPromptTemplate.from_template("""
    From this query:
    {query}

    If today is {today}, give me the start_date and the end_date mentioned in the query.
    Write it in XML format (without explanation) like this:
    <start_date>YYYY-MM-DD</start_date>
    <end_date>YYYY-MM-DD</end_date>

    Only respond with the XML, no other text.
    """)

    # Current date for context
    today = datetime.now().strftime("%Y-%m-%d")

    # Create the chain
    date_chain = date_prompt | llm

    # Invoke the chain to get date range
    try:
        date_response = date_chain.invoke({"query": query, "today": today})

        # Parse the XML response
        try:
            # Create a valid XML document for parsing
            xml_str = f"<root>{date_response.content}</root>"
            root = ET.fromstring(xml_str)

            start_date = root.find('start_date').text
            end_date = root.find('end_date').text
            # print(f"START DATE: {start_date}")
            # print(f"END DATE: {end_date}")

            # Validate dates (simple check)
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')

            return {
                "stock_code": stock_code,
                "start_date": start_date,
                "end_date": end_date
            }
        except (ET.ParseError, AttributeError, ValueError) as e:
            # Fallback to default if XML parsing fails
            print(f"XML parsing error: {e}")
            print(f"LLM response: {date_response.content}")

            # Default to last 30 days
            end_date = datetime.now()
            start_date = end_date - datetime.timedelta(days=30)

            return {
                "stock_code": stock_code,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            }

    except Exception as e:
        # Handle any errors in LLM processing
        print(f"Error processing date with LLM: {str(e)}")

        # Default to last 30 days
        end_date = datetime.now()
        start_date = end_date - datetime.timedelta(days=30)

        return {
            "stock_code": stock_code,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d')
        }

@tool
def analyze_stock_data(query: str) -> Dict:
    """
    Analyze stock data by processing the query and fetching data using analyze_basic_trading_data.
    """
    # First process the query to get stock code and date range
    query_info = process_stock_query(query)
    if "error" in query_info:
        return {"content": query_info["error"]}

    # Use the existing analyze_basic_trading_data function
    from utils.basic_trading_tool import analyze_basic_trading_data
    result = analyze_basic_trading_data(
                query_info['stock_code'],
                query_info['start_date'],
                query_info['end_date']
            )
    return result
