import os
import shutil

def move_files_to_folders(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Only process files, skip directories
        if os.path.isfile(file_path):
            # Get the filename without extension
            file_name_without_ext = os.path.splitext(filename)[0]

            # Create a new folder named after the file
            new_folder_path = os.path.join(directory, file_name_without_ext)

            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            # Move the file into the newly created folder
            new_file_path = os.path.join(new_folder_path, filename)
            shutil.move(file_path, new_file_path)

            print(f"Moved {filename} to {new_folder_path}")

# Replace 'your_directory_path' with the path of the directory containing the files
directory_path = r"C:\Users\djoh596\github\MARL_GNN_DFJSP_Extension\data_test\FJSP-benchmarks-main\2a_Hurink_sdata"
move_files_to_folders(directory_path)
