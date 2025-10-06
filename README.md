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

1.  **Process a specific PDF from a directory:** This option will guide you through selecting a directory, specifying page numbers, and choosing a PDF file to process.
2.  **Analyze Transcriptions with Agent:** This option will start an interactive session with the analysis agent.
3.  **Exit:** Exits the program.

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

