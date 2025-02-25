from langchain_core.messages import HumanMessage
from typing import Dict
import json

from utils.tuntun_api import get_basic_trading_data

import os
os.environ["LLM_MAX_LENGTH"] = 16385

def analyze_basic_trading_data(sec_code: str, start_date: str, end_date: str) -> Dict:
    """Analyze basic trading data for a given stock over a specified time range"""

    # print(f"START DATE: {start_date}")
    # print(f"END DATE: {end_date}")

    # Fetch the basic trading data using the get_basic_trading_data function
    trading_data = get_basic_trading_data(sec_code, start_date, end_date)

    # Check if the data is empty or the API returned an error
    if not trading_data:
        return {"content": f"No trading data available for {sec_code} between {start_date} and {end_date}."}

    # Process the data and prepare the analysis
    analysis_content = f"Analysis of {sec_code} from {start_date} to {end_date}:\n"

    start_date_api = trading_data[-1]["transactionDate"]
    end_date_api = trading_data[0]["transactionDate"]
    # print(f"START DATE API: {start_date_api}")
    # print(f"END DATE API: {end_date_api}")

    # TODO: using tiktoken, break the loop if total token length > LLM_MAX_LENGTH
    for data in trading_data:
        transaction_date = data["transactionDate"]
        open_price = data["openPrice"]
        close_price = data["closePrice"]
        low_price = data["lowPrice"]
        high_price = data["highPrice"]
        change_amount = data["changeAmount"]
        change_percentage = data["changePercentage"]
        volume = data["volume"]
        value = data["value"]

        analysis_content += (
            f"\nDate: {transaction_date}\n"
            f"  Open Price: {open_price}\n"
            f"  Close Price: {close_price}\n"
            f"  Low Price: {low_price}\n"
            f"  High Price: {high_price}\n"
            f"  Change: {change_amount} ({change_percentage}%)\n"
            f"  Volume: {volume}\n"
            f"  Total Value: {value}\n"
        )

    return {
        "content": analysis_content,
        "references": []  # In this case, no external references
    }
