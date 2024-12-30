import os

folder_path = "IdleCW"      # original
output_folder_path = "testaug2"         # new folder

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.startswith("testfile"):
        file_path = os.path.join(folder_path, file_name)
        output_file_path = os.path.join(output_folder_path, file_name)

        # Read the contents of the file
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Augment the data in each line
        augmented_lines = []
        for line in lines:
            data_points = line.strip().split()

            # Increase the value of specific data points
            augmented_points = []
            for i, point in enumerate(data_points):
                if i % 3 == 0 and float(point) != -1:  # Positions 3Y where Y = 0, 1, 2, ...
                    point = str(float(point) + 4)
                augmented_points.append(point)

            augmented_line = " ".join(augmented_points)
            augmented_lines.append(augmented_line)

        # Write the augmented data to the output file in the new folder
        with open(output_file_path, "w") as file:
            file.write("\n".join(augmented_lines))
