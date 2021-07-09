from os import listdir
from os.path import isfile, join
import random
import shutil
import time

darknet = True

if darknet:
    amtFiles = int(len([f for f in listdir("data") if isfile(join("data", f))])/2)
    
    validAmt = 0
    testAmt = 0
    for i in range(amtFiles):
        swap = random.randint(0, amtFiles - 1)
        swap2 = random.randint(0, amtFiles - 1)
        if swap != swap2:
            shutil.move("data/" + str(swap) + ".txt", "data/temp.txt")
            shutil.move("data/" + str(swap) + ".jpg", "data/temp.jpg")

            shutil.move("data/" + str(swap2) + ".txt", "data/" + str(swap) + ".txt")
            shutil.move("data/" + str(swap2) + ".jpg", "data/" + str(swap) + ".jpg")

            shutil.move("data/temp.txt", "data/" + str(swap2) + ".txt")
            shutil.move("data/temp.jpg", "data/" + str(swap2) + ".jpg")
            time.sleep(1/1000000.0)
