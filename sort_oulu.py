import os
import shutil

# Define your source and destination directories
source_dir = './data/video/train_oulu'
dest_dir = './data/video/train_oulu/'
dir_name = {"A":"Angry","D":"Disgust","F":"Fear","H":"Happy","S1":"Surprise","S2":"Sadness"}
# Get list of all files in source directory
files = os.listdir(source_dir)

# Loop through all files
for file in files:
    # Split the filename by spaces and get the last element
    ext = file.split(".")[-1]
    if ext in ["mp4", "avi", "mov"]:
        folder_name = file.split("_")[-1]
        folder_name = folder_name.split(".")[0]
        name = dir_name[folder_name]
        # Create a new directory path using the last element of the split filename
        new_dir = os.path.join(dest_dir, name)

        # If the directory doesn't exist, create it
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        # Move the file to the new directory
        shutil.move(os.path.join(source_dir, file), os.path.join(new_dir, file))