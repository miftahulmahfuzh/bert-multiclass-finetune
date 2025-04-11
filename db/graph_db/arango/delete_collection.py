# last edited on 17-03-2025. the script still failed to run

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

# Delete chat_logs collection
def delete_chat_logs_collection(db):
    try:
        # Check if the collection exists first
        if db.hasCollection("chat_logs"):
            # Delete the collection
            db._drop("chat_logs")
            print("Collection 'chat_logs' deleted successfully.")
            return True
        else:
            print("Collection 'chat_logs' does not exist.")
            return False
    except Exception as e:
        print(f"Failed to delete 'chat_logs' collection: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Connect to ArangoDB
    db = connect_to_arangodb()

    if db:
        # Delete the chat_logs collection
        delete_chat_logs_collection(db)
