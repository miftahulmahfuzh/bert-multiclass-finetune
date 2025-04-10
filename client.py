# Create server parameters for stdio connection
import environments
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1000)

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["math_server.py"],
)

m = "what's the weather in Jakarta today?"
m = "what's (3 + 5) x 12?"

async def run_agent():
    global m
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": m})

            for m in agent_response["messages"]:
                m.pretty_print()

if __name__ == "__main__":
    asyncio.run(run_agent())
