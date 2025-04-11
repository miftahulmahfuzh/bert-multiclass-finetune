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
        return db
    except Exception as e:
        print(f"Failed to connect to ArangoDB: {e}")
        return None

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

# Insert a sample document to verify the structure
def insert_sample_document(collection):
    # now = datetime.datetime.now(datetime.UTC).isoformat()  # Use timezone-aware datetime
    jakarta_tz = datetime.timezone(datetime.timedelta(hours=7))
    now = datetime.datetime.now(jakarta_tz).isoformat()
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

def update_document(collection, doc_id, update_dict):
    """
    Update specific fields in a document with values from a dictionary.

    Args:
        collection: The ArangoDB collection object
        doc_id: The document ID to update (e.g., 'chat_logs/173482')
        update_dict: Dictionary containing fields and values to update

    Returns:
        Updated document or None if document wasn't found
    """
    try:
        # Get the document by its ID
        doc = collection[doc_id]

        # Update only the fields provided in the update_dict
        for key, value in update_dict.items():
            if key in doc:
                doc[key] = value
            else:
                print(f"Warning: Field '{key}' does not exist in document {doc_id}")

        # Save the updated document
        doc.save()
        print(f"Document {doc_id} updated successfully")
        return doc

    except KeyError:
        print(f"Error: Document with ID {doc_id} not found")
        return None
    except Exception as e:
        print(f"Error updating document {doc_id}: {e}")
        return None

# Main function to initialize the logging table
def main():
    # Connect to ArangoDB
    db = connect_to_arangodb()
    if not db:
        print("Exiting due to connection failure.")
        return

    # Set up the logging collection
    # logs_collection = setup_logging_collection(db)

    # Insert a sample document to verify the structure
    # insert_sample_document(logs_collection)
    # print("Logging table initialization complete.")

    # update data
    logs_collection = db["chat_logs"]
    update_data = {
        "reaction": 0,  # Change reaction to UNLIKE
        "reaction_timestamp": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat()
    }

    # Update the document
    doc_id = "173482"
    updated_doc = update_document(logs_collection, doc_id, update_data)

    if updated_doc:
        print("Updated document:", updated_doc)


if __name__ == "__main__":
    main()
