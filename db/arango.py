from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from pyArango.connection import Connection
from pyArango.collection import Collection
from pyArango.document import Document
from db.utils import get_current_timestamp
from config import settings


class GraphDB(ABC):
    """Abstract base class for graph database operations."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def create_collection(self, collection_name: str) -> Any:
        """Create a new collection in the database."""
        pass

    @abstractmethod
    def get_collection(self, collection_name: str) -> Any:
        """Retrieve a collection by name."""
        pass

    @abstractmethod
    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> Any:
        """Insert a document into a collection."""
        pass

    @abstractmethod
    def update_document(self, collection_name: str, doc_id: str, update_data: Dict[str, Any]) -> Any:
        """Update a document in a collection."""
        pass


class PyArangoDB(GraphDB):
    """Implementation of GraphDB for ArangoDB using pyArango."""

    def __init__(self,
            url: str = "http://localhost:8529",
            username: str = "root",
            password: str = "",
            database: str = "_system"):
        """
        Initialize PyArangoDB with connection parameters.

        Args:
            url: The URL of the ArangoDB server
            username: ArangoDB username
            password: ArangoDB password
            database: Database name to use
        """
        self.url = url
        self.username = username
        self.password = password
        self.database_name = database
        self.connection = None
        self.db = None

    def connect(self) -> bool:
        """
        Establish connection to ArangoDB.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self.connection = Connection(
                arangoURL=self.url,
                username=self.username,
                password=self.password
            )
            self.db = self.connection[self.database_name]
            return True
        except Exception as e:
            print(f"Failed to connect to ArangoDB: {e}")
            return False

    def create_collection(self, collection_name: str) -> Collection:
        """
        Create a new collection if it doesn't exist.

        Args:
            collection_name: Name of the collection to create

        Returns:
            Collection object
        """
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")

        if not self.db.hasCollection(collection_name):
            collection = self.db.createCollection(name=collection_name)
            print(f"Collection '{collection_name}' created successfully.")
        else:
            collection = self.db[collection_name]
            print(f"Collection '{collection_name}' already exists.")

        return collection

    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Get a collection by name.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            Collection object or None if not found
        """
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")

        if self.db.hasCollection(collection_name):
            return self.db[collection_name]
        else:
            print(f"Collection '{collection_name}' does not exist.")
            return None

    def insert_document(self, collection_name: str, document: Dict[str, Any]) -> Optional[Document]:
        """
        Insert a document into a collection.

        Args:
            collection_name: Name of the collection
            document: Dictionary with document data

        Returns:
            Document object or None if insertion failed
        """
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")

        try:
            collection = self.get_collection(collection_name)
            if not collection:
                collection = self.create_collection(collection_name)

            doc = collection.createDocument(document)
            doc.save()
            print(f"Document inserted with ID: {doc['_id']}")
            return doc
        except Exception as e:
            print(f"Error inserting document: {e}")
            return None

    def update_document(self, collection_name: str, doc_id: str, update_data: Dict[str, Any]) -> Optional[Document]:
        """
        Update specific fields in a document.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID to update
            update_data: Dictionary with fields to update

        Returns:
            Updated document or None if update failed
        """
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")

        try:
            collection = self.get_collection(collection_name)
            if not collection:
                print(f"Collection '{collection_name}' does not exist.")
                return None

            # Get the document by its ID
            doc = collection[doc_id]

            # Update only the fields provided in the update_data
            for key, value in update_data.items():
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

    def create_chat_log(self,
                    user_id: int,
                    query_id: int,
                    channel: int,
                    user_query: str,
                    final_input: str,
                    final_output: str,
                    reaction: int,
                    selected_tools: List[str],
                    user_query_timestamp: str,
                    final_input_timestamp: str,
                    final_output_timestamp: str,
                    reaction_timestamp: str,
                    selected_tools_timestamp: str
                ) -> Optional[Document]:
        """
        Create a chat log document with default timestamps.

        Args:
            user_id: User identifier
            query_id: Query identifier
            channel: Channel identifier
            user_query: Original user query
            final_input: Processed input (defaults to user_query if None)
            final_output: System response
            reaction: user's feedback to ai's response (0: dislike, 1: like)
            selected_tools: List of tools used for processing
            user_query_timestamp: the timestamp when user_query is received in the system
            final_input_timestamp: the timestamp when final_input is generated by ai
            final_output_timestamp: the timestamp when final_output is generated by ai
            reaction_timestamp: the timestamp when user_reaction is received in the system
            selected_tools_timestamp: the timestamp when the chatbot finished selecting_tools

        Returns:
            Created document or None if creation failed
        """

        log_data = {
            "user_id": user_id,
            "query_id": query_id,
            "channel": channel,
            "user_query": user_query,
            "final_input": final_input,
            "final_output": final_output,
            "reaction": reaction,
            "selected_tools": selected_tools,
            "user_query_timestamp": user_query_timestamp,
            "final_input_timestamp": final_input_timestamp,
            "final_output_timestamp": final_output_timestamp,
            "reaction_timestamp": reaction_timestamp,
            "selected_tools_timestamp": selected_tools_timestamp
        }

        return self.insert_document(settings.LOG_DB_COLLECTION_NAME, log_data)

    def add_reaction(self, doc_id: str, reaction: int) -> Optional[Document]:
        """
        Add or update reaction to a chat log.

        Args:
            doc_id: Document ID
            reaction: Reaction value (1 for LIKE, 0 for UNLIKE)

        Returns:
            Updated document or None if update failed
        """
        update_data = {
            "reaction": reaction,
            "reaction_timestamp": get_current_timestamp()
        }

        return self.update_document(settings.LOG_DB_COLLECTION_NAME, doc_id, update_data)


def main():
    # Example usage
    db = PyArangoDB(
        url="http://localhost:8529",
        username="root",
        password="tuntun123",  # Replace with actual password
        database="tuntun_chatbot"
    )

    if db.connect():
        print("Connected to ArangoDB successfully")

        # Create collection if needed
        # db.create_collection("chat_logs")

        # # Create a sample chat log
        # now = get_current_timestamp()
        # doc = db.create_chat_log(
        #     user_id=101,
        #     query_id=1001,
        #     channel=1,
        #     user_query="What is the weather?",
        #     final_input="User asked: What is the weather?",
        #     final_output="The weather is sunny!",
        #     reaction=1,
        #     selected_tools=["weather_api", "stock_price"],
        #     user_query_timestamp=now,
        #     final_input_timestamp=now,
        #     final_output_timestamp=now,
        #     reaction_timestamp=now,
        #     selected_tools_timestamp=now
        # )

        # Update a document with a reaction
        updated_doc = db.add_reaction("179154", 0)  # UNLIKE

        if updated_doc:
            print("Updated document:", updated_doc)
    else:
        print("Failed to connect to ArangoDB")


if __name__ == "__main__":
    main()
