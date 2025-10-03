from google_ocr_2 import transcribe_document_google_ocr
from config import PROJECT_ID, LOCATION, PROCESSOR_ID

def transcribe_document(filepath, output_dir):
    """Transcribes a document using Google Cloud Document AI."""
    transcribe_document_google_ocr(
        project_id=PROJECT_ID,
        location=LOCATION,
        processor_id=PROCESSOR_ID,
        file_path=filepath,
        mime_type='application/pdf',  # Assuming all files are PDFs for now
        output_file_path=output_dir
    )