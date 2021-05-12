from os import listdir
from os.path import isfile, join
import random
import shutil

validNum = round(len([f for f in listdir("data") if isfile(join("data", f))]) * .2)
testNum = round(len([f for f in listdir("data") if isfile(join("data", f))]) * .1)

for i in range(validNum):
    filelist = [f for f in listdir("data") if isfile(join("data", f))]
    filename = filelist[random.randint(0, len(filelist) - 1)]
    shutil.move("data/" + filename, "valid/" + filename)
    
for i in range(testNum):
    filelist = [f for f in listdir("data") if isfile(join("data", f))]
    filename = filelist[random.randint(0, len(filelist) - 1)]
    shutil.move("data/" + filename, "test/" + filename)
    
filelist = [f for f in listdir("data") if isfile(join("data", f))]
for i in range(len(filelist)):
    shutil.move("data/" + filelist[i], "train/" + filelist[i])