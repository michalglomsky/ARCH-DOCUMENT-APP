import os
import json

# Load existing config or create a default one
CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {
        "TRANSCRIPTION_MODEL": "gemini-1.5-flash",
        "EDITING_MODEL": "gemini-1.5-flash",
        "PROJECT_DIR": os.path.dirname(os.path.abspath(__file__)),
        "SEARCH_DIR": os.path.join(os.path.dirname(os.path.abspath(__file__)), "DOKUMENTY SAMPLE"),
        "PROCESSED_PDFS_DIR": os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_pdfs"),
        "TRANSCRIPTIONS_DIR": os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_pdfs", "transcriptions"),
        "EDITED_DIR": os.path.join(os.path.dirname(os.path.abspath(__file__)), "edited")
    }

# Load configuration
config = load_config()

# Ensure directories exist
os.makedirs(config["PROCESSED_PDFS_DIR"], exist_ok=True)
os.makedirs(config["TRANSCRIPTIONS_DIR"], exist_ok=True)
os.makedirs(config["EDITED_DIR"], exist_ok=True)

# Shared variables
TRANSCRIPTION_MODEL = config["TRANSCRIPTION_MODEL"]
EDITING_MODEL = config["EDITING_MODEL"]
PROJECT_DIR = config["PROJECT_DIR"]
SEARCH_DIR = config["SEARCH_DIR"]
PROCESSED_PDFS_DIR = config["PROCESSED_PDFS_DIR"]
TRANSCRIPTIONS_DIR = config["TRANSCRIPTIONS_DIR"]
EDITED_DIR = config["EDITED_DIR"]