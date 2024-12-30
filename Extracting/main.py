import os

with open('kpts_3d.dat', 'r') as old:
    with open('extracted', 'w') as new:
        lines = old.readlines()
        c = 0
        for line in lines:
            if line[0] == 'R':
                new.writelines(str(c) + ' ' + line[10:])
            else:
                c += 1

with open('extracted', 'r') as ext:
    with open('extracted_nums_0', 'w') as ext_nums:
        lines = ext.readlines()
        n = 129
        a = 0
        for line in lines:
            sp = line.index(' ')
            c = int(line[:sp])
            if c == n:
                    ext_nums.writelines(str(a) + line[sp:])
            else:
                n = c
                a += 1
                #with open('extracted_nums_'+str(a), 'w') as ext_nums:
                ext_nums.writelines('\n')
                ext_nums.writelines(str(a) + line[sp:])

def makeNewFile(lines, c):
    folderName = 'IdleZin'
    #os.makedirs(folderName)
    with open(folderName+'/testfile' + str(c+14), 'w') as newf:
        newf.writelines(lines)

def split_file_by_space(f):
    lines = f.readlines()
    c = 0
    newLines = []
    for line in lines:
        if line == '\n':
            makeNewFile(newLines, c)
            newLines.clear()
            c += 1
        else:
            newLines.append(line[2:])


with open('extracted_nums_0', 'r') as f:
   split_file_by_space(f)