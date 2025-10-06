# ARCH DOCUMENT APP

**DISCLAIMER:** This project is currently under active development. Features may be incomplete or subject to change.

## About The Project

This project is a command-line interface (CLI) application designed to streamline the processing and analysis of architectural documents. It automates the extraction of specific pages from PDF documents, performs Optical Character Recognition (OCR) using Google Cloud Document AI, and provides an interactive agent for data analysis.

## Features

*   **PDF Page Extraction:** Extracts specific pages or page ranges from PDF files.
*   **OCR Transcription:** Uses Google Cloud Document AI for high-quality text transcription from images and PDFs.
*   **Interactive Analysis Agent:** An interactive agent, built with `langchain` and `langgraph`, to analyze the transcribed documents.
*   **Versioned Output:** Automatically versions output files to prevent overwriting existing data.

## Getting Started

### Prerequisites

*   Python 3.x
*   Google Cloud SDK authenticated with a project that has the Document AI API enabled.
*   A `config.json` file in the root directory with the following format:
    ```json
    {
      "PROJECT_ID": "your-gcp-project-id",
      "LOCATION": "your-gcp-project-location",
      "PROCESSOR_ID": "your-document-ai-processor-id"
    }
    ```

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/your_username/ARCH-DOCUMENT-APP.git
    ```
2.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

Run the main application from the command line:

```sh
python main.py
```

The application will present a menu with the following options:

### 1. Process a specific PDF from a directory

This option allows you to perform the core document processing workflow. When you select this option, the application will prompt you for the following information:

*   **Directory to search for PDFs:** The root directory where the application will search for your PDF files.
*   **Page numbers to extract:** You can specify a single page (e.g., `1`), a comma-separated list of pages (e.g., `1,3,5`), or a range of pages (e.g., `5-7`).
*   **Name of the PDF file to process:** The name of the PDF file you want to process (e.g., `wniosek.pdf`).

The application will then:

1.  Search for the specified PDF file in the given directory and its subdirectories.
2.  For each found PDF, it will extract the specified pages into a new, processed PDF file. This new file will be saved in the `output/processed_pdfs` directory with a unique name based on the original file name and its parent directory.
3.  The extracted PDF is then sent to the Google Cloud Document AI API for OCR transcription.
4.  The transcribed text is saved to a `.txt` file in the `output/transcription_output_dir` directory.

### 2. Analyze Transcriptions with Agent

This option launches an interactive chat session with an AI agent. The agent is designed to help you analyze the transcribed documents. You can ask the agent questions about the content of the transcriptions, and it will use its language understanding capabilities to provide answers. The agent is built using `langchain` and `langgraph`, and it can work with the transcribed data in a structured way.

To exit the agent, type `finished`.

### 3. Exit

This option exits the program.


## Dependencies

The project uses the following major dependencies:

*   `pytesseract`
*   `langchain`
*   `langgraph`
*   `groq`
*   `Pillow`
*   `python-dotenv`
*   `torch`
*   `transformers`
*   `accelerate`
*   `google-cloud-documentai`
*   `ipython`
*   `tabulate`
*   `protobuf`
*   `pandas`
*   `PyMuPDF`
*   `langchain-groq`

