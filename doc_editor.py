import google.generativeai as genai
from config import EDITING_MODEL

def edit_document(prompt, document_text):
    """Edits a document using a generative model."""
    model = genai.GenerativeModel(EDITING_MODEL)
    return model.generate_content([prompt, document_text])