import os
import shutil
from config import PROCESSED_PDFS_DIR

def process_directory_pdfs(search_dir):
    """Processes all PDFs in a directory, moving them to a new location with a modified name."""
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith(".pdf") and file.lower().startswith("wniosek"):
                original_path = os.path.join(root, file)
                dir_name = os.path.basename(os.path.dirname(original_path))
                new_filename = f"{os.path.splitext(file)[0]}_{dir_name}.pdf"
                new_filepath = os.path.join(PROCESSED_PDFS_DIR, new_filename)
                
                if not os.path.exists(new_filepath):
                    shutil.copy(original_path, new_filepath)
                    print(f"Copied and renamed '{original_path}' to '{new_filepath}'")