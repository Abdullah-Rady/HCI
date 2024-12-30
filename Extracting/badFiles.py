import os

directory = 'IdleACW/'

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    with open(filepath, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)
        if line_count < 10 or line_count > 30:
            print(f'{filename}: {line_count} lines')


# Rotation:
    # testfile138: 9 lines
    # testfile107: 6 lines
    # testfile34: 8 lines
    # testfile0: 0 lines
    # testfile139: 8 lines
    # testfile128: 3 lines
#ZoomIn:
    # testfile68: 9 lines
    # testfile125: 9 lines
    # testfile69: 8 lines
    # testfile0: 0 lines
    # testfile123: 9 lines
    # testfile70: 9 lines

# ZoomOut:
    # testfile67: 3 lines
    # testfile0: 0 lines
    # testfile112: 9 lines

#AntiClockWise:
    # testfile57: 5
    # lines
    # testfile49: 8
    # lines
    # testfile0: 0
    # lines