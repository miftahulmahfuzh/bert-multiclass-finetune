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
        self.counters_collection = "system_counters"  # Collection to store auto-increment counters
        self.channel_collection = "channel_info"  # Collection to store the last channel from each user

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

            # Initialize counters collection if it doesn't exist
            self._init_counters_collection()

            # Initialize channel info collection if it doesn't exist
            self._init_channel_collection()

            return True
        except Exception as e:
            print(f"Failed to connect to ArangoDB: {e}")
            return False

    def _init_counters_collection(self) -> None:
        """Initialize the counters collection if it doesn't exist."""
        if not self.db.hasCollection(self.counters_collection):
            collection = self.db.createCollection(name=self.counters_collection)

            # Create a counter document for query_id
            counter_doc = collection.createDocument({
                "_key": "query_id_counter",
                "name": "query_id",
                "current_value": 0
            })
            counter_doc.save()
            print(f"Created {self.counters_collection} collection with query_id counter.")
        else:
            # Ensure query_id counter exists
            collection = self.db[self.counters_collection]
            try:
                collection["query_id_counter"]
            except KeyError:
                counter_doc = collection.createDocument({
                    "_key": "query_id_counter",
                    "name": "query_id",
                    "current_value": 0
                })
                counter_doc.save()
                print("Created query_id counter document.")

    def _init_channel_collection(self) -> None:
        """Initialize the channel info collection if it doesn't exist."""
        if not self.db.hasCollection(self.channel_collection):
            collection = self.db.createCollection(name=self.channel_collection)
            print(f"Created {self.channel_collection} collection.")

    def get_next_query_id(self) -> int:
        """
        Get the next available query_id (auto-increment).

        Returns:
            int: The next available query_id
        """
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")

        try:
            collection = self.db[self.counters_collection]
            counter_doc = collection["query_id_counter"]

            # Increment the current value
            current_value = counter_doc["current_value"]
            next_value = current_value + 1

            # Update the counter in the database
            counter_doc["current_value"] = next_value
            counter_doc.save()

            return next_value
        except Exception as e:
            print(f"Error getting next query_id: {e}")
            # In case of error, return a default value that indicates an issue
            return -1

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
                # return None
                return "failed"

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
            # return doc
            return "success"

        except KeyError:
            print(f"Error: Document with ID {doc_id} not found")
            # return None
            return "failed"
        except Exception as e:
            print(f"Error updating document {doc_id}: {e}")
            # return None
            return "failed"

    def get_channel(self, user_id: int) -> int:
        """
        Get the channel for a specific user from channel_info collection.

        Args:
            user_id: User identifier

        Returns:
            int: Channel identifier or 0 if not found
        """
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")

        try:
            collection = self.get_collection(self.channel_collection)
            if not collection:
                collection = self.create_collection(self.channel_collection)
                print(f"Created {self.channel_collection} collection.")
                return 0

            # Try to find document with user_id as key
            user_key = str(user_id)
            try:
                doc = collection[user_key]
                return int(doc["channel"])
            except KeyError:
                print(f"No channel found for user_id {user_id}")
                return 0

        except Exception as e:
            print(f"Error getting channel for user_id {user_id}: {e}")
            return 0

    def update_channel(self, user_id: int, channel: int) -> Optional[Document]:
        """
        Update or create channel information for a user.

        Args:
            user_id: User identifier
            channel: Channel identifier

        Returns:
            Document object or None if operation failed
        """
        result = {"status":"failed", "message":"connection to database not established"}
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")

        try:
            collection = self.get_collection(self.channel_collection)
            if not collection:
                collection = self.create_collection(self.channel_collection)

            # Use user_id as the document key for unique constraint
            user_key = str(user_id)

            # Check if the user already exists
            try:
                # User exists, update the channel
                doc = collection[user_key]
                doc["channel"] = str(channel)
                doc["updated_at"] = get_current_timestamp()
                doc.save()
                message = f"Updated channel to '{channel}' for user_id {user_id}"
                result = {"status": "success", "message": message}
                print(message)
                return result
            except Exception as e:
                # User doesn't exist, create new document
                doc = collection.createDocument({
                    "_key": user_key,
                    "user_id": str(user_id),
                    "channel": str(channel),
                    "created_at": get_current_timestamp(),
                    "updated_at": get_current_timestamp()
                })
                doc.save()
                message = f"Created new channel record for user_id {user_id} with channel {channel}"
                result = {"status": "success", "message": message}
                print(message)
                return result

        except Exception as e:
            result["message"] = f"Error updating channel for user_id {user_id}: {e}"
            return result

    def create_chat_log(self,
                    user_id: int,
                    user_query: str,
                    final_input: str,
                    final_output: str,
                    reaction: int,
                    selected_tools: List[str],
                    user_query_timestamp: str,
                    final_input_timestamp: str,
                    final_output_timestamp: str,
                    reaction_timestamp: str,
                    selected_tools_timestamp: str,
                    query_id: Optional[int] = None,
                    channel: Optional[int] = None
                ) -> Optional[Document]:
        """
        Create a chat log document with auto-incrementing query_id if not provided.
        Get channel from channel_info collection if not provided.

        Args:
            user_id: User identifier
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
            query_id: Optional query identifier (auto-generated if None)
            channel: Optional channel identifier (fetched from channel_info if None)

        Returns:
            Created document or None if creation failed
        """
        # If query_id is not provided, get the next available ID
        if query_id is None:
            query_id = self.get_next_query_id()
            if query_id == -1:
                print("Failed to generate query_id. Using timestamp as fallback.")
                # Fallback to using timestamp as a unique ID
                query_id = int(user_query_timestamp.replace('.', ''))

        # If channel is not provided, get it from channel_info collection
        if channel is None:
            channel = self.get_channel(user_id)

        log_data = {
            "user_id": str(user_id),
            "query_id": str(query_id),
            "channel": str(channel),
            "user_query": user_query,
            "final_input": final_input,
            "final_output": final_output,
            "reaction": str(reaction),
            "selected_tools": selected_tools,
            "user_query_timestamp": user_query_timestamp,
            "final_input_timestamp": final_input_timestamp,
            "final_output_timestamp": final_output_timestamp,
            "reaction_timestamp": reaction_timestamp,
            "selected_tools_timestamp": selected_tools_timestamp
        }

        return self.insert_document(settings.LOG_DB_COLLECTION_NAME, log_data)

    def update_reaction(self, query_id: str, reaction: str, user_id: Optional[str] = None) -> Optional[Document]:
        """
        Add or update reaction to a chat log based on query_id.

        Args:
            query_id: Query identifier for the chat log
            reaction: Reaction value (1 for LIKE, 0 for UNLIKE)
            user_id: Optional user identifier to ensure the correct document is updated
                (in case multiple documents have the same query_id)

        Returns:
            Updated document or None if update failed
        """
        result = {"status":"failed", "message":"connection to database not established"}
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")

        try:
            collection = self.get_collection(settings.LOG_DB_COLLECTION_NAME)
            if not collection:
                print(f"Collection '{settings.LOG_DB_COLLECTION_NAME}' does not exist.")
                return None

            # Build the AQL query to find the document by query_id
            aql_query = "FOR doc IN @@collection FILTER doc.query_id == @query_id"

            # Add user_id filter if provided
            if user_id:
                aql_query += " AND doc.user_id == @user_id"

            aql_query += " RETURN doc"

            # Prepare bind variables for the query
            bind_vars = {
                "@collection": settings.LOG_DB_COLLECTION_NAME,
                "query_id": str(query_id)
            }

            if user_id:
                bind_vars["user_id"] = str(user_id)

            # Execute the query
            result = self.db.AQLQuery(aql_query, bindVars=bind_vars, rawResults=True)

            if not result:
                print(f"No document found with query_id {query_id}")
                return None

            # Get the first matching document
            doc_data = result[0]
            doc_id = doc_data["_key"]

            # Update the document with new reaction data
            update_data = {
                "reaction": reaction,
                "reaction_timestamp": get_current_timestamp()
            }

            status = self.update_document(settings.LOG_DB_COLLECTION_NAME, doc_id, update_data)
            if status == "success":
                message = f"Updated reaction to '{reaction}' for query_id {query_id}"
                result = {"status": status, "message": message}
            else:
                message = "failed updating reaction for query_id {query_id}"
            return result

        except Exception as e:
            print(f"Error updating reaction for query_id {query_id}: {e}")
            return None

    def get_chat_history(self, user_id: int, n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get chat history for a specific user, sorted from oldest to newest.
        Optionally limit to the n most recent entries.

        Args:
            user_id: User identifier
            n: Optional limit on number of history items to return (newest n items)

        Returns:
            List of dictionaries containing chat history information
        """
        result = {"status":"failed", "message":"connection to database not established"}
        if not self.db:
            raise ConnectionError("Database connection not established. Call connect() first.")
            result = {"status":"failed", "message": msg}
            return result

        try:
            # Build the AQL query to find all documents for the user, sorted by timestamp
            aql_query = """
            FOR doc IN @@collection
            FILTER doc.user_id == @user_id
            """

            # If n is specified, get the newest n items (sort DESC, limit, then sort ASC)
            if n is not None and n > 0:
                aql_query += """
                SORT doc.user_query_timestamp DESC
                LIMIT @limit
                SORT doc.user_query_timestamp ASC
                """
            else:
                # Otherwise, simply sort by timestamp (oldest first)
                aql_query += """
                SORT doc.user_query_timestamp ASC
                """

            aql_query += """
            RETURN {
                "user_query": doc.user_query,
                "final_output": doc.final_output,
                "reaction": doc.reaction,
                "user_query_timestamp": doc.user_query_timestamp,
                "final_output_timestamp": doc.final_output_timestamp
            }
            """

            # Prepare bind variables for the query
            bind_vars = {
                "@collection": settings.LOG_DB_COLLECTION_NAME,
                "user_id": str(user_id)
            }

            # Add limit parameter if n is specified
            if n is not None and n > 0:
                bind_vars["limit"] = n

            # Execute the query
            result = self.db.AQLQuery(aql_query, bindVars=bind_vars, rawResults=True)

            # Convert the result to a list
            history = list(result)

            msg = f"Retrieved {len(history)} history records for user_id {user_id}"
            if n is not None and n > 0:
                msg += f" (limited to {n} entries)"
            print(msg)
            result = {"status": "success", "message": msg, "items": history}
            return result

        except Exception as e:
            msg = f"Error retrieving chat history for user_id {user_id}: {e}"
            result = {"status":"failed", "message": msg}
            return result


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

        # Example of updating a user's channel
        db.update_channel(user_id="101", channel="2")

        # Retrieve the channel for a user
        channel = db.get_channel(user_id="101")
        print(f"User 101 is using channel {channel}")

        # Create a sample chat log with auto-incremented query_id and channel from channel_info
        now = get_current_timestamp()
        doc = db.create_chat_log(
            user_id="101",
            user_query="What is the weather?",
            final_input="User asked: What is the weather?",
            final_output="The weather is sunny!",
            reaction="like",
            selected_tools=["weather_api", "stock_price"],
            user_query_timestamp=now,
            final_input_timestamp=now,
            final_output_timestamp=now,
            reaction_timestamp=now,
            selected_tools_timestamp=now
            # query_id not provided - will be auto-generated
            # channel not provided - will be fetched from channel_info
        )

        if doc:
            print(f"Created chat log with query_id: {doc['query_id']} and channel: {doc['channel']}")

            # Example of updating a document's reaction using query_id
            query_id = doc["query_id"]
            new_reaction = "0"
            updated_doc = db.update_reaction(query_id=query_id, reaction=new_reaction)  # UNLIKE
            if updated_doc:
                print(f"Updated reaction for query_id {query_id} to '{new_reaction}'")

            # Create another log to demonstrate incremental query_id
            # Update user's channel first
            db.update_channel(user_id="102", channel="3")

            doc2 = db.create_chat_log(
                user_id="102",
                user_query="What time is it?",
                final_input="User asked: What time is it?",
                final_output="The current time is 3:00 PM.",
                reaction="dislike",
                selected_tools=["time_api"],
                user_query_timestamp=now,
                final_input_timestamp=now,
                final_output_timestamp=now,
                reaction_timestamp=now,
                selected_tools_timestamp=now
                # Channel will be retrieved from channel_info
            )

            if doc2:
                print(f"Created second chat log with query_id: {doc2['query_id']} and channel: {doc2['channel']}")

        # Example of updating a document with a reaction
        # updated_doc = db.update_reaction("179154", 0)  # UNLIKE
        # if updated_doc:
        #     print("Updated document:", updated_doc)
    else:
        print("Failed to connect to ArangoDB")


if __name__ == "__main__":
    main()
