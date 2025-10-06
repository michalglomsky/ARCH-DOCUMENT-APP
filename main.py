import os
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from google_ocr_2 import transcribe_document_google_ocr
import fitz  # PyMuPDF
from src.react_agent.run_agent import run_agent

import sys
import google.cloud.documentai


load_dotenv()



# Load Google Cloud configuration
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    PROJECT_ID = config["PROJECT_ID"]
    LOCATION = config["LOCATION"]
    PROCESSOR_ID = config["PROCESSOR_ID"]
    
    print("Configuration loaded successfully from config.json")

except FileNotFoundError:
    print("Error: config.json not found. Please create it.")
    exit()
except KeyError as e:
    print(f"Error: Missing key in config.json: {e}")
    exit()

# --- Global State ---
transcribed_files = {}

# --- Core Functions ---

def get_versioned_filepath(directory, filename, extension):
    """Generates a versioned filepath to avoid overwriting files."""
    version = 1
    while True:
        filepath = os.path.join(directory, f"{filename}_v{version}.{extension}")
        if not os.path.exists(filepath):
            return filepath
        version += 1

def transcribe_document(filepath, output_dir=None, output_filename=None):
    """Performs OCR on a document using Google Cloud Document AI."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at '{filepath}'")
        return

    extension = os.path.splitext(filepath)[1].lower()

    if filepath in transcribed_files:
        print("This file has been transcribed before. Do you want to proceed and create a new transcription? (y/n)")
        if input("> ").lower() != 'y':
            return

    try:
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp',
            '.gif': 'image/gif'
        }
        
        mime_type = mime_type_map.get(extension)
        
        if not mime_type:
            print(f"Error: Unsupported file type '{extension}'. Please provide a supported image or PDF file.")
            return

        transcription_dir = output_dir
        os.makedirs(transcription_dir, exist_ok=True)
        
        # \textbf{MODIFICATION: Use provided filename or generate a versioned one}
        if output_filename:
            output_filepath = os.path.join(transcription_dir, output_filename)
        else:
            # Fallback to old behavior for other use cases
            filename = os.path.splitext(os.path.basename(filepath))[0]
            output_filepath = get_versioned_filepath(transcription_dir, f"{filename}_transcription", "txt")


        transcribe_document_google_ocr(
            project_id=PROJECT_ID,
            location=LOCATION,
            processor_id=PROCESSOR_ID,
            file_path=filepath,
            mime_type=mime_type,
            output_file_path=output_filepath,
        )

        transcribed_files[filepath] = output_filepath
        print(f"Transcription saved to: {output_filepath}")

    except Exception as e:
        print(f"An error occurred during transcription: {e}")


def process_directory_pdfs(search_dir, pages_to_extract_str, file_name):
    """
    Searches for PDF files with a specific name in a directory, 
    extracts specified pages, and then transcribes the extracted pages.
    """
    output_dir = "output"
    processed_pdfs_dir = os.path.join(output_dir, "processed_pdfs")
    transcription_output_dir = os.path.join(output_dir, "transcription_output_dir")

    os.makedirs(processed_pdfs_dir, exist_ok=True)
    os.makedirs(transcription_output_dir, exist_ok=True)

    # Find all PDF files in the directory that match file_name
    pdf_files = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file == file_name:
                pdf_files.append(os.path.join(root, file))

    if not pdf_files:
        print(f"No PDF files named '{file_name}' found in '{search_dir}'.")
        return

    # Parse page numbers
    try:
        pages_to_extract = []
        for part in pages_to_extract_str.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                pages_to_extract.extend(range(start - 1, end))
            else:
                pages_to_extract.append(int(part) - 1)
    except ValueError:
        print("Error: Invalid page numbers format. Use comma-separated numbers and ranges (e.g., 1,3,5-7).")
        return

    for pdf_path in pdf_files:
        pdf_filename = os.path.basename(pdf_path)
        try:
            # \textbf{MODIFICATION: Extract ID and create new filenames}
            parent_dir_name = os.path.basename(os.path.dirname(pdf_path))
            try:
                # Get the string after the last underscore
                doc_id = parent_dir_name.rsplit('_', 1)[1]
            except IndexError:
                print(f"Warning: Could not find ID in parent directory name '{parent_dir_name}'. Using 'NO_ID'.")
                doc_id = "NO_ID"

            base_filename = os.path.splitext(pdf_filename)[0]

            # Construct new filenames based on the required format
            new_pdf_filename = f"{base_filename}_{doc_id}_processed.pdf"
            transcription_filename = f"{base_filename}_{doc_id}_transcribed.txt"
            
            extracted_pdf_path = os.path.join(processed_pdfs_dir, new_pdf_filename)

            # Extract pages and create a new PDF
            doc = fitz.open(pdf_path)
            new_doc = fitz.open()
            for page_num in pages_to_extract:
                if 0 <= page_num < doc.page_count:
                    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                else:
                    print(f"Warning: Page {page_num + 1} is out of range for the document '{pdf_filename}'.")

            if new_doc.page_count > 0:
                new_doc.save(extracted_pdf_path)
                print(f"Extracted pages saved to: {extracted_pdf_path}")

                # \textbf{MODIFICATION: Call transcribe_document with the specific output filename}
                transcribe_document(
                    extracted_pdf_path, 
                    output_dir=transcription_output_dir, 
                    output_filename=transcription_filename
                )
            else:
                print(f"No pages were extracted from {pdf_filename}.")

            new_doc.close()
            doc.close()

        except Exception as e:
            print(f"An error occurred while processing {pdf_filename}: {e}")


# --- Main Application Loop ---

"""
Opcje są uzupełnione w mainie choice == 3 for development purposes
"""

def main():
    """Main function to run the CLI application."""
    while True:
        print("\n--- Document Transcription & Editing CLI ---")
        print("What would you like to do?")
        
        print("1. Process a specific PDF from a directory")
        print("2. Analyze Transcriptions with Agent") # New option
        print("3. Exit") # Was 4

        choice = input("> ")

        if choice == '1':
            print("Enter the directory to search for PDFs:")
            search_dir = r"C:\Users\micha\Desktop\ARCH DOCUMENT APP\DOKUMENTY SAMPLE" # input("> ").strip('"')
            print("Enter the page numbers to extract (e.g., 1,3,5-7):")
            pages_to_extract = "1" # input("> ")
            print("Enter the name of the PDF file to process (e.g., wniosek.pdf):")
            file_name = "wniosek.pdf" # input("> ").strip('"')
            process_directory_pdfs(search_dir, pages_to_extract, file_name)

        elif choice == '2':
            run_agent()

        elif choice == '3':
            print("Exiting the program.")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2 or 3.")

if __name__ == "__main__":
    main()