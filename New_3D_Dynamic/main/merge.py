import os


def merge_text_files(input_folder, output_filename):

    with open(output_filename, 'w') as output_file:

        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as input_file:
                output_file.write(input_file.read())

# Example usage
input_folder = os.path.join(os.getcwd(), "main/IdleMouse Mar 5")
output_filename = "main/merged_output.txt"
merge_text_files(input_folder, output_filename)