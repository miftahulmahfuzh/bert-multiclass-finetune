import environments
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o")

m = """
answer both of these questions:
    1. what's (3 + 5) x 12?
    2. what is the weather in nyc?
"""

async def run_agent():
    global m
    async with MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                # Make sure to update to the full absolute path to your math_server.py file
                "args": ["/home/devmiftahul/nlp/faq_chatbot/news_mcp_server2/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # make sure you start your weather server on port 8000
                "url": "http://localhost:8838/sse",
                "transport": "sse",
            }
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())

        # math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        # for m in math_response["messages"]:
        #     m.pretty_print()

        # weather_response = await agent.ainvoke({"messages": "what is the weather in nyc?"})
        # for m in weather_response["messages"]:
        #     m.pretty_print()

        math_weather_response = await agent.ainvoke({"messages": m})
        for m in math_weather_response["messages"]:
            m.pretty_print()


if __name__ == "__main__":
    asyncio.run(run_agent())
