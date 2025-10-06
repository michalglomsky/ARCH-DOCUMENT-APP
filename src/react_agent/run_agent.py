import pandas as pd
from .graph import create_agent_graph
from .prompts import system_prompt
from langchain_core.messages import SystemMessage, HumanMessage

MAX_HISTORY_TURNS = 10 # Keep the last 10 messages + system prompt

def run_agent():
    """Initializes and runs the langgraph agent."""
    graph = create_agent_graph()
    
    messages = [SystemMessage(content=system_prompt)]
    
    print("Agent started. Type 'finished' to quit.")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'finished':
            break
        
        messages.append(HumanMessage(content=user_input))

        # Trim history if it gets too long
        if len(messages) > MAX_HISTORY_TURNS + 1:
            messages = [messages[0]] + messages[-(MAX_HISTORY_TURNS):]
        
        result = graph.invoke({"messages": messages})
        
        messages = result['messages']
        response_message = messages[-1]

        # Check if the agent wants to show the dataset
        if response_message.tool_calls and response_message.tool_calls[0].name == 'show_dataset':
            df_as_list = result.get('dataframe', [])
            df = pd.DataFrame(df_as_list)
            if df.empty:
                print("The dataset is empty.")
            else:
                print("Here is the current dataset:")
                print(df.to_markdown(index=False))
            # Append the tool output message so the agent knows it was displayed
            messages.append(response_message.tool_calls[0].to_tool_message("Displayed the dataset."))
        else:
            print(response_message.content)