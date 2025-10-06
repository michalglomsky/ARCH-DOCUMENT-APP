import os
import pandas as pd
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import json

# Construct the path to config.json relative to this file
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config.json'))

# Load the configuration file
with open(config_path, 'r') as f:
    config = json.load(f)

# Get the transcription directory from the config
TRANSCRIPTION_DIR = config['directories']['transcription_dir']

@tool
def list_transcriptions() -> list[str]:
    """Lists all available transcription files."""
    if not os.path.exists(TRANSCRIPTION_DIR):
        return ["Transcription directory not found."]
    files = [f for f in os.listdir(TRANSCRIPTION_DIR) if f.endswith(".txt")]
    if not files:
        return ["No transcription files found."]
    return files

@tool
def read_transcription(filename: str) -> str:
    """Reads the content of a specific transcription file."""
    filepath = os.path.join(TRANSCRIPTION_DIR, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            if not content:
                return "The file is empty."
            return content
    except FileNotFoundError:
        return f"File not found: {filename}"

@tool
def extract_personal_info(transcription: str) -> str:
    """
    Extracts personal information (name, surname, etc.) from a given transcription.
    Returns a JSON string with the extracted information.
    """
    chat = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
    
    system = "You are an expert at extracting personal information from documents. Extract the name, surname, and any other relevant personal details from the provided text. Return the information as a JSON object."
    human = f"Please extract the personal information from the following transcription:\n\n{transcription}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    
    chain = prompt | chat
    response = chain.invoke({"transcription": transcription})
    return response.content

@tool
def add_to_dataframe(data: str, df: list[dict]) -> list:
    """Adds the extracted data to the pandas DataFrame."""
    import json
    import pandas as pd
    new_data = json.loads(data)
    df = pd.DataFrame(df)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    return df.to_dict('records')

@tool
def show_dataset() -> str:
    """Call this to show the current dataset to the user. This will display the data that has been collected so far."""
    # This tool is a signal. The actual display logic will be handled by the caller.
    return "Displayed the dataset."

@tool
def update_dataframe(index: int, column: str, value: str, df: list[dict]) -> list:
    """Updates a specific cell in the DataFrame."""
    import pandas as pd
    df = pd.DataFrame(df)
    df.at[index, column] = value
    return df.to_dict('records')

@tool
def mark_file_as_processed(filename: str, processed_files: list[str]) -> list[str]:
    """Marks a file as processed to avoid processing it again. Use this after a file has been successfully read, extracted, and added to the dataset."""
    if filename not in processed_files:
        processed_files.append(filename)
    return processed_files
