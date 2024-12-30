import os

folder_path = "IdleZout"

# Initialize a list to store the names of files with more than 63 data points
files_with_more_than_63 = []

# Iterate over the files in the folder
for i in range(83):  # X ranges from 0 to 113
    file_name = f"testfile{i}"
    file_path = os.path.join(folder_path, file_name)

    # Check if the file exists
    if os.path.isfile(file_path):
        # print('jfggf')
        with open(file_path, "r") as file:
            # Read each line in the file
            for line in file:
                # Split the line by whitespace to get the data points
                data_points = line.split()

                # Check if the line contains more than 63 data points
                if len(data_points) > 63:
                    files_with_more_than_63.append(file_name)
                    break  # Move to the next file

# Print the names of files with more than 63 data points
for file_name in files_with_more_than_63:
    print(file_name)
