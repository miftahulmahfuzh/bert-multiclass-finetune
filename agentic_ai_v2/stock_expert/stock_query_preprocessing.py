from datetime import datetime, timedelta
from langchain_core.tools import tool
from typing import Dict, Tuple
import re
from utils.basic_trading_tool import analyze_basic_trading_data

@tool
def process_stock_query(query: str) -> Dict:
    """
    Process a natural language query about stock data to extract stock code and date range.
    Handles relative date ranges like 'last 7 days' and specific date ranges.
    """
    # Extract stock code - looking for capital letters possibly followed by numbers
    stock_match = re.search(r'([A-Z]+\d*)', query)
    if not stock_match:
        return {"error": "No valid stock code found in query"}

    stock_code = stock_match.group(1)

    # Handle relative date ranges
    if 'last' in query.lower():
        # Extract number of days
        days_match = re.search(r'last (\d+) days?', query.lower())
        if days_match:
            days = int(days_match.group(1))
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Format dates as strings
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')

            return {
                "stock_code": stock_code,
                "start_date": start_date_str,
                "end_date": end_date_str
            }

    # Handle specific date ranges (if needed)
    # Example: "BBCA from 2024-03-01 to 2024-03-07"
    date_range_match = re.search(r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})', query)
    if date_range_match:
        return {
            "stock_code": stock_code,
            "start_date": date_range_match.group(1),
            "end_date": date_range_match.group(2)
        }

    # Default to last 7 days if no specific range is mentioned
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
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

    # Construct the query string for analyze_basic_trading_data
    formatted_query = f"{query_info['stock_code']} from {query_info['start_date']} to {query_info['end_date']}"

    # Use the existing analyze_basic_trading_data function
    # return analyze_basic_trading_data(formatted_query)
    result = analyze_basic_trading_data(
                query_info['stock_code'],
                query_info['start_date'],
                query_info['end_date']
            )
    return result
