from pyArango.connection import Connection
import pandas as pd
import os
from datetime import datetime

def connect_to_arangodb():
    try:
        conn = Connection(
            arangoURL="http://localhost:8529",
            username="root",
            password="tuntun123"
        )
        db = conn["_system"]  # Use the default '_system' database
        return conn, db
    except Exception as e:
        print(f"Failed to connect to ArangoDB: {e}")
        return None, None

def fetch_date_range_records(conn, db, start_date, end_date):
    collection_name = "chat_logs"

    # Check if the collection exists
    if not db.hasCollection(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return [], ""

    # Make sure end_date is inclusive by setting it to the end of the day
    end_date_inclusive = f"{end_date}T23:59:59"

    # Construct the AQL query to find documents within date range
    aql_query = """
    FOR doc IN @@collection
        FILTER doc.user_query_timestamp >= @start_date AND doc.user_query_timestamp <= @end_date_inclusive
        RETURN {
            "query_id": doc.query_id,
            "user_id": doc.user_id,
            "user_query": doc.user_query,
            "final_input": doc.final_input,
            "final_output": doc.final_output,
            "timestamp": doc.user_query_timestamp
        }
    """

    # Define the bind variables
    bind_vars = {
        "@collection": collection_name,
        "start_date": start_date,
        "end_date_inclusive": end_date_inclusive
    }

    try:
        # Execute the query
        result = db.AQLQuery(aql_query, bindVars=bind_vars, rawResults=True)

        # Create a full query string with bind variables for documentation
        full_query = aql_query.replace("@@collection", f'"{collection_name}"')
        full_query = full_query.replace("@start_date", f'"{start_date}"')
        full_query = full_query.replace("@end_date_inclusive", f'"{end_date_inclusive}"')

        return list(result), full_query
    except Exception as e:
        print(f"Query execution failed: {e}")
        return [], ""

def save_to_excel(results, query_text, file_path, results_sheet_name, query_sheet_name):
    # Create a pandas DataFrame from the results
    results_df = pd.DataFrame(results)

    # Create a DataFrame for the query
    query_df = pd.DataFrame([{"Query": query_text}])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save both DataFrames to different sheets in the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name=results_sheet_name, index=False)
        query_df.to_excel(writer, sheet_name=query_sheet_name, index=False)

    print(f"Results saved to {file_path}")

def main():
    # Set the date range to fetch
    start_date = "2025-03-01"
    end_date = "2025-03-17"

    # Connect to ArangoDB
    conn, db = connect_to_arangodb()
    if not db:
        print("Exiting due to connection failure.")
        return

    # Fetch records within the date range
    results, query_text = fetch_date_range_records(conn, db, start_date, end_date)

    # Generate timestamp for the filename
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the Excel file path with timestamp and date range
    excel_path = f"xlsx/fetch_timestamp_from_{start_date}_to_{end_date}_{datetime_str}.xlsx"

    # Print summary of the results
    if results:
        print(f"Found {len(results)} records from {start_date} to {end_date}.")

        # Save the results and query to Excel
        save_to_excel(
            results,
            query_text,
            excel_path,
            "fetch_result",
            "query_info"
        )
    else:
        print(f"No records found from {start_date} to {end_date}.")

        # Save empty results with query info
        save_to_excel(
            [],
            query_text,
            excel_path,
            "fetch_result",
            "query_info"
        )

if __name__ == "__main__":
    main()
