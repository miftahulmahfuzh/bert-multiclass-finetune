from pyArango.connection import Connection
import pandas as pd
import os
from datetime import datetime, time

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

def fetch_today_records(conn, db):
    collection_name = "chat_logs"

    # Check if the collection exists
    if not db.hasCollection(collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return [], ""

    # Get today's date in ISO format (YYYY-MM-DD)
    today_date = datetime.now().date().isoformat()

    # Construct the AQL query to find documents from today
    aql_query = """
    FOR doc IN @@collection
        FILTER STARTS_WITH(doc.user_query_timestamp, @today_date)
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
        "today_date": today_date
    }

    try:
        # Execute the query
        result = db.AQLQuery(aql_query, bindVars=bind_vars, rawResults=True)

        # Create a full query string with bind variables for documentation
        full_query = aql_query.replace("@@collection", f'"{collection_name}"')
        full_query = full_query.replace("@today_date", f'"{today_date}"')

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
    # Connect to ArangoDB
    conn, db = connect_to_arangodb()
    if not db:
        print("Exiting due to connection failure.")
        return

    # Fetch today's records
    results, query_text = fetch_today_records(conn, db)

    # Generate timestamp for the filename
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the Excel file path with timestamp
    excel_path = f"xlsx/fetch_timestamp_{datetime_str}.xlsx"

    # Print summary of the results
    if results:
        print(f"Found {len(results)} records from today.")

        # Save the results and query to Excel
        save_to_excel(
            results,
            query_text,
            excel_path,
            "fetch_result",
            "query_info"
        )
    else:
        print("No records found from today.")

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
