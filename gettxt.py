import os

# Define the directory to scan
dir_path = "./data/face/test_ravdess"

# Define the output file path
output_file_path = "output.txt"

# Open the output file in write mode
with open(output_file_path, "w") as f:
    # Walk through the directory
    for root, dirs, files in os.walk(dir_path):
        # Skip if no files in the directory
        if not files:
            continue
        # Get the parent directory name as the label
        label = os.path.basename(os.path.dirname(root))
        # Write the directory path (relative to dir_path) and label to the output file
        relative_path = os.path.relpath(root, dir_path)
        relative_path = relative_path.replace("\\", "/")
        f.write(f"{relative_path} {label}\n")