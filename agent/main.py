"""The main entry point for the agent.

It defines the workflow graph, state, tools, nodes and edges.
"""
import asyncio
from typing import TypedDict

from copilotkit import CopilotKitState
from copilotkit.langgraph import copilotkit_emit_state
from langchain.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt


class Searches(TypedDict):
    query: str
    done: bool


class AgentState(CopilotKitState):
    agent_name: str
    proverbs: list[str]
    searches: list[Searches]


@tool
def get_weather(location: str):
    """Get the weather for a given location."""
    return f"The weather for {location} is 70 degrees."


# Extract tool names from backend_tools for comparison
backend_tools = [get_weather]
backend_tool_names = [tool.name for tool in backend_tools]


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[str]:
    if "searches" not in state:
        state["searches"] = []

    # We can call copilotkit_emit_state to emit updated state
    # before a node finishes
    await copilotkit_emit_state(config, state)

    # Simulate state updates
    for search in state["searches"]:
        await asyncio.sleep(1)
        search["done"] = True

        # We can also emit updates in a loop to simulate progress
        await copilotkit_emit_state(config, state)

    # if not state.get("agent_name"):
    #     # Interrupt and wait for the user to respond with a name
    #     state["agent_name"] = interrupt(
    #         "Before we start, what would you like to call me?"
    #     )

    # 1. Define the model
    model = ChatOpenAI(model="gpt-5-mini")

    # 2. Bind the tools to the model
    model_with_tools = model.bind_tools(
        [*state.get("copilotkit", {}).get("actions", []), *backend_tools],
        parallel_tool_calls=False,
    )

    # 3. Define the system message by which the chat model will be run
    system_message = SystemMessage(
        content=f"You are a helpful assistant. The current proverbs are {state.get('proverbs', [])}."
    )

    # 4. Run the model to generate a response
    response = await model_with_tools.ainvoke(
        [system_message, *state["messages"]], config
    )

    # only route to tool node if tool is not in the tools list
    if route_to_tool_node(response):
        print("routing to tool node")
        return Command(goto="tool_node", update={"messages": [response]})

    # 5. We've handled all tool calls, so we can end the graph.
    return Command(goto=END, update={"messages": [response]})


def route_to_tool_node(response: BaseMessage):
    """Route to tool node if any tool call in the response matches a backend tool name."""
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        if tool_call.get("name") in backend_tool_names:
            return True
    return False


# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=backend_tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

graph = workflow.compile()
