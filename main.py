import os
from config import SEARCH_DIR
from pdf_processor import process_directory_pdfs
from transcription import transcribe_document
from doc_editor import edit_document

def main():
    """Main function to run the CLI application."""
    while True:
        print("\n--- Document Transcription & Editing CLI ---")
        print("What would you like to do?")
        print("1. Transcribe a document")
        print("2. Edit a document")
        print("3. Process PDFs from a directory")
        print("4. Exit")

        choice = input("> ")

        if choice == '1':
            print("Enter the path to the document you want to transcribe:")
            filepath = input("> ").strip('"')
            print(f"Transcribing {filepath}...")
            # This is a placeholder for the actual transcription call
            # You will need to adapt the new transcribe_document function to work with the CLI
            # For example, you might need to handle file uploads or specify paths differently
            # transcribe_document(filepath)

        elif choice == '2':
            print("Enter the path to the document you want to edit:")
            filepath = input("> ").strip('"')
            print("Enter the editing instructions:")
            instructions = input("> ")
            # This is a placeholder for the actual editing call
            # You will need to adapt the new edit_document function to work with the CLI
            # edit_document(filepath, instructions)

        elif choice == '3':
            print(f"Processing PDFs from: {SEARCH_DIR}")
            # Assuming process_directory_pdfs now only needs SEARCH_DIR
            process_directory_pdfs(SEARCH_DIR)

        elif choice == '4':
            print("Exiting the program.")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()