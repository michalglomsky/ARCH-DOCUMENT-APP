@tool
def mark_file_as_processed(filename: str, processed_files: list[str]) -> list[str]:
    """Marks a file as processed to avoid processing it again. Use this after a file has been successfully read, extracted, and added to the dataset."""
    if filename not in processed_files:
        processed_files.append(filename)
    return processed_files