from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from typing import Literal
from langchain_core.tools import tool
from IPython.display import Image, display
from dotenv import load_dotenv
import asyncio
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")
    
tools = [get_weather]

graph = create_react_agent(model, tools=tools)

# This line will only work in an interactive environment like a Jupyter notebook
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

async def main():
    """Main function to interact with the agent in a loop."""
    print("\n--- Simple ReAct Agent CLI ---")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        inputs = {
            "messages": [
                ("system", "You are a helpful assistant. You only have access to one tool: `get_weather`. Do not use any other tools. If you are asked a question you cannot answer with the available tools, say you cannot answer."),
                HumanMessage(content=user_input)
            ]
        }
        
        print("Agent: ", end="")
        async for event in graph.astream_events(inputs, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # On new LLM token, print it out
                    print(content, end="")
            elif kind == "on_tool_end":
                # On tool end, print a newline for better formatting
                print("\n")

        print("\n")

if __name__ == "__main__":
    # This block will only run when the script is executed directly
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")