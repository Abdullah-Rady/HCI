
# This script processes a file containing data and saves the recordings to separate files.
def process_and_save_data(file_path, output_folder, num_files):
    with open(file_path, 'r') as file:
        lines = file.readlines()


    # Separate recordings and non-recordings
    recordings = [group[10:-1] for group in lines if group.startswith("Recording")]
    # non_recordings = [group for group in grouped_lines if not group[0].startswith("Recording")]

    # Group lines by 5
    grouped_lines = [recordings[i:i+5] for i in range(0, len(recordings), 5)]

    # Save recordings to new files
    for i, recording_group in enumerate(grouped_lines):
        
        if len(recording_group) - 1 == i:
            continue

        recording_filename = f"{i + num_files}.txt"
        recording_filepath = os.path.join(output_folder, action_name, recording_filename)

        # os.makedirs(recording_filepath, exist_ok=True)  # Create directories if not exists

        with open(recording_filepath, 'w') as recording_file:
            recording_file.writelines(recording_group)


if __name__ == "__main__":
    import os

    file_path = os.getcwd() + '/main/merged_output.txt'
    output_folder = os.getcwd() + '/main/data'
    action_name = 'Idle'

    os.makedirs(os.path.join(output_folder , action_name), exist_ok=True)


    files = os.listdir(os.path.join(output_folder, action_name))
    num_files = len(files)
    # Create output folder if it doesn't exist
    # os.makedirs(output_folder, exist_ok=True)


    process_and_save_data(file_path, output_folder, num_files)
