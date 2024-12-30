import os

# f1 = 'ZOut'
# f2 = 'ZIn'
# f3 = 'Rot'

f1 = 'ClockWiseRotation'
f2 = 'ZoomIn'
f3 = 'ZoomOut'

os.makedirs(f1)
os.makedirs(f2)
os.makedirs(f3)


def fix(folder, newfolder):
    files = os.listdir(folder)
    for file in files:
        with open(folder+'/'+file, 'r') as inFile,\
             open(newfolder+'/'+file, 'w') as outFile:
            for line in inFile:
                outFile.write(line[2:])

if __name__ == '__main__':
    fix('RotationCSemiRaw', f1)
    fix('ZoomInSemiRaw', f2)
    fix('ZoomOutSemiRaw', f3)