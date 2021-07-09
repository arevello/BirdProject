from os import listdir
from os.path import isfile, join
import random
import shutil

darknet = True

if darknet:
    amtFiles = int(len([f for f in listdir("data") if isfile(join("data", f))])/2)

    print(amtFiles)
    startAmt = 0
    for i in range(amtFiles):
        shutil.move("data/" + str(startAmt) + ".txt", "data/" + str(startAmt) + ".txt")
        shutil.move("data/" + str(startAmt) + ".jpg", "data/" + str(startAmt) + ".jpg")
        startAmt += 1
