import os
from google.cloud import documentai
from google.api_core.client_options import ClientOptions

def transcribe_document_google_ocr(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str, output_file_path: str) -> None:
    """Transcribes a document using Google Cloud Document AI."""
    try:
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        print("----------------",location,"-----------------")
        name = client.processor_path(project_id, location, processor_id)

        with open(file_path, "rb") as image:
            image_content = image.read()

        raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=name, raw_document=raw_document)

        result = client.process_document(request=request)
        document = result.document

        # Save the transcribed text to the specified output file
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(document.text)

        print(f"Transcription successful. Output saved to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred during Google OCR transcription: {e}")
