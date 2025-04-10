import environments
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from messages import m

# model = ChatOpenAI(model="gpt-4o")
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)

async def run_agent():
    global m
    async with MultiServerMCPClient(
        {
            "stocks": {
                "command": "python",
                "args": ["/home/devmiftahul/nlp/faq_chatbot/news_mcp_server2/news_server.py"],
                "transport": "stdio",
            },
            "math": {
                "command": "python",
                "args": ["/home/devmiftahul/nlp/faq_chatbot/news_mcp_server2/math_server.py"],
                "transport": "stdio",
            },
        }
    ) as client:
        agent = create_react_agent(model, client.get_tools())

        agent_response = await agent.ainvoke({"messages": m})
        for m in agent_response["messages"]:
            m.pretty_print()


if __name__ == "__main__":
    asyncio.run(run_agent())
