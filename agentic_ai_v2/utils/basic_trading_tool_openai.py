from langchain_core.messages import HumanMessage
from typing import Dict
import json
from utils.tuntun_api import get_basic_trading_data
import os
import tiktoken

def analyze_basic_trading_data(sec_code: str, start_date: str, end_date: str) -> Dict:
    """
    Analyze basic trading data for a given stock over a specified time range.
    Limits the amount of data based on token count to prevent context length exceeded errors.
    """
    # Fetch the basic trading data using the get_basic_trading_data function
    trading_data = get_basic_trading_data(sec_code, start_date, end_date)

    # Check if the data is empty or the API returned an error
    if not trading_data:
        return {"content": f"No trading data available for {sec_code} between {start_date} and {end_date}."}

    # Get max token length from environment or use a default value
    # Subtract a buffer for other parts of the conversation
    max_tokens = int(os.environ.get("LLM_MAX_LENGTH", 16385)) - 5000

    # Initialize the tokenizer
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Prepare the initial content
    analysis_content = f"Analysis of {sec_code} from {start_date} to {end_date}:\n\n"

    # Record the date range from the API response
    start_date_api = trading_data[-1]["transactionDate"]
    end_date_api = trading_data[0]["transactionDate"]

    # Calculate tokens for the initial content
    initial_token_count = len(encoding.encode(analysis_content))
    current_token_count = initial_token_count

    # Prepare a summary section to provide context even if we truncate data
    summary_section = (
        f"Summary statistics:\n"
        f"- First trading day in data: {start_date_api}\n"
        f"- Last trading day in data: {end_date_api}\n"
        f"- Number of trading days: {len(trading_data)}\n"
        f"- Starting price: {trading_data[-1]['openPrice']}\n"
        f"- Ending price: {trading_data[0]['closePrice']}\n"
        f"- Overall change: {trading_data[0]['closePrice'] - trading_data[-1]['openPrice']}\n\n"
    )

    analysis_content += summary_section
    current_token_count += len(encoding.encode(summary_section))

    # Track how many days we include
    days_included = 0
    truncated = False

    # Process the data and add to analysis, stopping if we hit token limit
    analysis_details = ""
    for data in trading_data:
        day_content = (
            f"Date: {data['transactionDate']}\n"
            f"  Open Price: {data['openPrice']}\n"
            f"  Close Price: {data['closePrice']}\n"
            f"  Low Price: {data['lowPrice']}\n"
            f"  High Price: {data['highPrice']}\n"
            f"  Change: {data['changeAmount']} ({data['changePercentage']}%)\n"
            f"  Volume: {data['volume']}\n"
            f"  Total Value: {data['value']}\n\n"
        )

        # Check token count
        day_token_count = len(encoding.encode(day_content))
        if current_token_count + day_token_count > max_tokens:
            truncated = True
            break

        analysis_details += day_content
        current_token_count += day_token_count
        days_included += 1

    # Add a notice if we truncated the data
    if truncated:
        truncation_notice = (
            f"\n[Note: Data has been truncated to fit within token limits. "
            f"Showing {days_included} out of {len(trading_data)} trading days.]\n"
        )
        analysis_content += truncation_notice

    # Add the detailed days data
    analysis_content += analysis_details

    # Return the combined analysis
    return {
        "content": analysis_content,
        "references": []  # No external references in this case
    }
