from tuntun_chatbot_v2 import rag_chain
import json
import time
import ast

if __name__ == "__main__":
    # Import gradio
    import gradio as gr

    # Define the API endpoint function that will be exposed
    def api_chat(message, user_id="101"):
        """
        This function will be exposed as an API endpoint for streaming responses
        """
        # Stream the response chunks
        # for chunk in rag_chain(question=message, user_id=user_id, stream=True):
        #     yield chunk
        result_str = rag_chain(question=message, user_id=user_id, stream=False)
        result = ast.literal_eval(result_str)
        query_id = result["query_id"]
        response = result["final_output"]
        print(f"QUERY_ID: {query_id}")
        print(f"RESPONSE: {response}")

        first_chunk = json.dumps({"query_id": query_id})
        yield first_chunk

        # Then yield the content character by character
        for char in response:
            time.sleep(0.005)
            yield char


    # Create the UI with Blocks
    with gr.Blocks() as demo:
        with gr.Row():
            user_id_input = gr.Textbox(label="User ID", value="101", lines=1)
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Message")
        clear = gr.Button("Clear")

        # Store chat history in state
        state = gr.State([])

        def user(user_message, history, user_id):
            # Add user message to history
            history.append([user_message, None])
            return "", history, history

        def bot(history, user_id):
            # Get the last user message
            user_message = history[-1][0]
            # Use a placeholder to progressively update
            history[-1][1] = ""
            query_id = None

            # Call the rag_chain function with the message and user_id
            for chunk in rag_chain(question=user_message, user_id=user_id, stream=True):
                # If this is the first chunk, it might contain the query_id
                if not query_id and chunk.startswith("{"):
                    try:
                        data = json.loads(chunk)
                        query_id = data.get("query_id")
                        continue  # Skip adding this to the visible output
                    except json.JSONDecodeError:
                        # Not JSON, treat as regular content
                        history[-1][1] += chunk
                else:
                    # Regular content chunk
                    history[-1][1] += chunk

                # Yield the current state to update the UI
                yield history

            return history

        def clear_chat():
            return [], []

        # Set up the message processing pipeline
        msg.submit(user, [msg, chatbot, user_id_input], [msg, chatbot, state]).then(
            bot, [state, user_id_input], [chatbot]
        )
        clear.click(clear_chat, None, [chatbot, state])

    # Create the API endpoint specifically for streaming
    api = gr.Interface(
        fn=api_chat,
        inputs=[
            gr.Textbox(label="Message"),
            gr.Textbox(label="User ID", value="101")
        ],
        outputs=gr.Textbox(),
        title="Chat API",
        description="API endpoint for chat with user_id parameter",
        allow_flagging="never"
    )

    # Launch both the UI and API
    gr.TabbedInterface(
        [demo, api],
        ["Chat UI", "API"]
    ).launch(server_name='0.0.0.0')
