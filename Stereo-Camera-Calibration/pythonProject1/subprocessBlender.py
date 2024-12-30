import subprocess

blender_path = "/snap/blender/3486/blender"
script_path = "/home/decatrox/Blender projects/rot45.py"

subprocess.call([blender_path, "-b", "-P", script_path])
