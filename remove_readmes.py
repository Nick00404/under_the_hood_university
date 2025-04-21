import os

# Define the root directory where the folders are
root_directory = r"C:\Users\admin\Videos\Under_the_hood_university"

# Walk through all the subdirectories and files
for dirpath, dirnames, filenames in os.walk(root_directory):
    # Check if README.md exists in this directory
    if 'README.md' in filenames:
        # Get the full path of README.md
        readme_path = os.path.join(dirpath, 'README.md')
        try:
            # Remove the README.md file
            os.remove(readme_path)
            print(f"Removed: {readme_path}")
        except Exception as e:
            print(f"Error removing {readme_path}: {e}")
