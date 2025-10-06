import pandas as pd
from .graph import create_agent_graph
from .prompts import system_prompt
from langchain_core.messages import SystemMessage, HumanMessage

def run_agent():
    """Initializes and runs the langgraph agent."""
    # CompiledStateGraph class
    graph = create_agent_graph()
    
    messages = [SystemMessage(content=system_prompt)]
    
    print("Agent started. Type 'finished' to quit.")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'finished':
            break
        
        messages.append(HumanMessage(content=user_input))
        
        result = graph.invoke({"messages": messages})
        
        response_message = result['messages'][-1]
        messages.append(response_message)
        
        print(response_message.content)