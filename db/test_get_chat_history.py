from db.arango import PyArangoDB
from db.utils import get_current_timestamp

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
        db.update_channel(user_id=101, channel=2)

        # Create multiple sample chat logs with timestamps increasing over time
        for i in range(1):
            # Wait a moment to ensure timestamps are different
            import time
            time.sleep(1)

            now = get_current_timestamp()
            doc = db.create_chat_log(
                user_id=101,
                user_query=f"Sample query {i+1}",
                final_input=f"User asked: Sample query {i+1}",
                final_output=f"Sample response {i+1}",
                reaction="none",  # Alternate between 0 and 1
                selected_tools=["sample_tool"],
                user_query_timestamp=now,
                final_input_timestamp=now,
                final_output_timestamp=now,
                reaction_timestamp=now,
                selected_tools_timestamp=now
            )

            if doc:
                print(f"Created chat log {i+1} with query_id: {doc['query_id']}")

        print("\nRetrieving all chat history for user 101:")
        result = db.get_chat_history(user_id=101)
        all_history = result["items"]
        for index, item in enumerate(all_history):
            print(f"Entry {index+1}: {item['user_query']} - Reaction: {item['reaction']}")

        # Testing with limit
        limit = 2
        print(f"\nRetrieving last {limit} history items for user 101:")
        result = db.get_chat_history(user_id=101, n=limit)
        limited_history = result["items"]
        for index, item in enumerate(limited_history):
            print(f"Entry {index+1}: {item['user_query']} - Reaction: {item['reaction']}")

        # Change the limit
        limit = 3
        print(f"\nRetrieving last {limit} history items for user 101:")
        result = db.get_chat_history(user_id=101, n=limit)
        limited_history = result["items"]
        for index, item in enumerate(limited_history):
            print(f"Entry {index+1}: {item['user_query']} - Reaction: {item['reaction']}")
    else:
        print("Failed to connect to ArangoDB")

if __name__ == "__main__":
    main()
