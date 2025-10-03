import os

def get_versioned_filepath(filepath, version_suffix='_v'):
    """Generates a versioned filepath if the file already exists."""
    if not os.path.exists(filepath):
        return filepath
    
    name, ext = os.path.splitext(filepath)
    version = 1
    while True:
        new_filepath = f"{name}{version_suffix}{version}{ext}"
        if not os.path.exists(new_filepath):
            return new_filepath
        version += 1