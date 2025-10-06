from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from .state import AgentState
from .tools import list_transcriptions, read_transcription, extract_personal_info, add_to_dataframe, display_dataframe, update_dataframe
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from .prompts import system_prompt

def create_agent_graph():
    """Creates the langgraph agent."""
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
    tools = [list_transcriptions, read_transcription, extract_personal_info, add_to_dataframe, display_dataframe, update_dataframe]
    
    llm_with_tools = llm.bind_tools(tools)

    # 1. Define the nodes
    def agent_node(state: AgentState):
        result = llm_with_tools.invoke(state['messages'])
        return {'messages': [result]}

    # Using prebuilt Class ToolNode
    tool_node = ToolNode(tools)

    # 2. Define the edges
    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        # If there are no tool calls, then we finish
        if not last_message.tool_calls:
            return END
        # Otherwise if there are tool calls, we call the tools
        return "tools"

    # 3. Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()