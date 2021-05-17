import os
import glob
datader = "./firedata"

listtxt = os.listdir(datader)

for file in glob.glob("firedata/*txt"):
    f = open(file, 'r') 
    data = f.readlines()
    new_Data = []
    for line in data:
        lines = line.split((' '))
        lines[0] = '0'
        new_Data.append(' '.join(lines))
    print(new_Data)
    f.close()
    f = open(file, 'w') 
    for line in new_Data:
        f.write(line)
    f.close()