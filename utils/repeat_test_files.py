import os
import shutil

data_path = ["MK1", "MK2", "MK3", "MK4", "MK5",
                          "MK6", "MK7", "MK8", "MK9", "MK10",
                          "MK11", "MK12", "MK13", "MK14", "MK15"]

for data in data_path:

        # Define the source folder and number of copies
        source_folder = f'../data_test/FJSP-benchmarks-main/1_Brandimarte/{data}'
        num_copies = 4 # Number of copies to create

        # Automatically find the source file that ends with '_001'
        for file_name in os.listdir(source_folder):
            if file_name.endswith('_001.fjs'):
                source_file = file_name
                break

        # Get the file name components and increment the last part
        base_name, ext = os.path.splitext(source_file)
        prefix, last_part = base_name.rsplit('_', 1)

        # Create copies and rename
        for i in range(1, num_copies + 1):
            new_file_name = f"{prefix}_{int(last_part) + i:03d}{ext}"
            source_path = os.path.join(source_folder, source_file)
            target_path = os.path.join(source_folder, new_file_name)
            shutil.copy(source_path, target_path)

        print(f"Created {num_copies} copies successfully!")
