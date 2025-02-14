import os
os.environ["TAVILY_API_KEY"] = ""

from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

from langchain_core.messages import HumanMessage

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

#####################################################################################

# from model.Model import llm
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1000,
    )

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import Literal

members = ["Researcher", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = ["FINISH"] + members

class routeResponse(BaseModel):
    next: Literal[*options]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))


def supervisor_agent(state):
    supervisor_chain = (
        prompt
        | llm.with_structured_output(routeResponse)
    )
    return supervisor_chain.invoke(state)

#####################################################################################

import os
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import ToolNode

os.environ["TAVILY_API_KEY"] = "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"

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()


@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    result = "\n\n".join(
        [f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>' for doc in docs])
    return result


research_tool_node = ToolNode([tavily_tool, scrape_webpages])

#####################################################################################

from langchain_community.tools import HumanInputRun
from langgraph.prebuilt import ToolNode


def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)


human_input_tool = HumanInputRun(input_func=get_input)
general_tool_node = ToolNode([human_input_tool])

#####################################################################################

import functools
import operator
from typing import Sequence, TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, SystemMessage

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

researcher_system_message = SystemMessage(
    content="You are a news researcher. Do not do any coding task. Respond with result from the tools only. Other agent will handle subsequent tasks. Do not perform any tasks designated for other agents.")
research_agent = create_react_agent(llm, tools=research_tool_node, state_modifier=researcher_system_message)
research_node = functools.partial(agent_node, agent=research_agent, name="assistant_researcher")

coder_system_message = SystemMessage(
    content="You are a coder. Use tool to ask human what programming language to use. Other agent will handle any non-coding tasks, do not perform any tasks designated for other agents. Focus strictly on delivering the required code in your response.")
# code_agent = create_react_agent(llm, tools=[python_repl_tool], state_modifier=coder_prompt)
code_agent = create_react_agent(llm, tools=general_tool_node, state_modifier=coder_system_message)
code_node = functools.partial(agent_node, agent=code_agent, name="assistant_coder")

workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_agent)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

for s in graph.stream(
    {
        "messages": [
            HumanMessage(content="Code hello world and print it to the terminal")
        ]
    }
):
    if "__end__" not in s:
        print(s)
        print("----")
