import os

# set the directory containing the files you want to count lines of
directory = "ZoomOutSemiRaw/"

total_lines = 0
num_files = 0

# loop through each file in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    # only count lines of regular files (i.e., not directories)
    if os.path.isfile(filepath):
        with open(filepath, "r") as file:
            # count the number of lines in the file
            num_lines = sum(1 for line in file)
            total_lines += num_lines
            num_files += 1

# calculate the average number of lines per file
if num_files > 0:
    avg_lines_per_file = total_lines / num_files
else:
    avg_lines_per_file = 0

print("Average number of lines per file:", avg_lines_per_file)

# Average number of lines per file: 23.938650306748468  Rotation
# Average number of lines per file: 18.378205128205128  ZoomIn
# Average number of lines per file: 14.284671532846716  ZoomOut

