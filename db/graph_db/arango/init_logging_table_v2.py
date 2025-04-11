from pyArango.connection import Connection
from pyArango.collection import Collection
from pyArango.document import Document
import datetime

# Connect to ArangoDB
def connect_to_arangodb():
    try:
        conn = Connection(
            arangoURL="http://localhost:8529",
            username="root",
            password="tuntun123"  # Replace with the password from docker-compose
        )
        db = conn["_system"]  # Use the default '_system' database
        return conn, db
    except Exception as e:
        print(f"Failed to connect to ArangoDB: {e}")
        return None, None

# Create or get the logging collection
def setup_logging_collection(db):
    collection_name = "chat_logs"

    # Check if the collection exists, if not create it
    if not db.hasCollection(collection_name):
        db.createCollection(name=collection_name)
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")

    return db[collection_name]

# Create indexes for faster queries using the correct method for pyArango 1.3.1
def create_indexes(conn, collection):
    # Define the list of fields to index
    timestamp_fields = [
        "final_input_timestamp",
        "final_output_timestamp",
        "reaction_timestamp",
        "selected_tools_timestamp",
        "user_query_timestamp"
    ]

    # Get existing indexes to avoid duplicates or conflicts
    try:
        existing_indexes = collection.getIndexes()
        print("Existing indexes:", existing_indexes)
        existing_index_names = []
        for idx_type in existing_indexes:
            for idx in existing_indexes[idx_type].values():
                if hasattr(idx, 'infos') and 'name' in idx.infos:
                    existing_index_names.append(idx.infos['name'])
    except Exception as e:
        print(f"Error getting indexes: {e}")
        existing_index_names = []

    for field in timestamp_fields:
        index_name = f"idx_{field}"
        # Check if the index already exists
        if index_name in existing_index_names:
            print(f"Index {index_name} already exists, skipping creation.")
            continue

        try:
            # Use the connection object to create the index via the REST API
            request = conn.session.post(
                f"http://localhost:8529/_db/_system/_api/index",
                json={
                    "type": "persistent",
                    "fields": [field],
                    "name": index_name,
                    "collection": "chat_logs"
                },
                auth=("root", "tuntun123")
            )

            if request.status_code in (201, 200):
                print(f"Index created on {field} with name {index_name}")
            else:
                print(f"Failed to create index on {field}: {request.text}")
        except Exception as e:
            print(f"Failed to create index on {field}: {e}")

# Insert a sample document to verify the structure
def insert_sample_document(collection):
    now = datetime.datetime.now(datetime.UTC).isoformat()  # Use timezone-aware datetime
    sample_log = {
        "channel": 1,  # int
        "final_input": "User asked: What is the weather?",  # str
        "final_input_timestamp": now,  # datetime
        "final_output": "The weather is sunny!",  # str
        "final_output_timestamp": now,  # datetime
        "query_id": 1001,  # int
        "reaction": 1,  # int (1 for LIKE, 0 for UNLIKE, can be empty)
        "reaction_timestamp": now,  # datetime
        "selected_tools": ["weather_api", "nlp_parser"],  # List[str]
        "selected_tools_timestamp": now,  # datetime
        "user_id": 101,  # int
        "user_query": "What is the weather?",  # str
        "user_query_timestamp": now  # datetime
    }

    doc = collection.createDocument(sample_log)
    doc.save()
    print(f"Sample document inserted with ID: {doc['_id']}")

# Main function to initialize the logging table
def main():
    # Connect to ArangoDB
    conn, db = connect_to_arangodb()
    if not db:
        print("Exiting due to connection failure.")
        return

    # Set up the logging collection
    logs_collection = setup_logging_collection(db)

    # Create indexes for better performance
    create_indexes(conn, logs_collection)

    # Insert a sample document to verify the structure
    insert_sample_document(logs_collection)

    print("Logging table initialization complete.")

if __name__ == "__main__":
    main()
