from os import listdir
from os.path import isfile, join
import random
import shutil

darknet = True

if darknet:
    amtFiles = int(len([f for f in listdir("data") if isfile(join("data", f))])/2)
    validNum = round(amtFiles * .2)
    testNum = round(amtFiles * .1)
    print(validNum, testNum)
    
    validAmt = 0
    testAmt = 0
    for i in range(amtFiles):
        imgType = random.randint(0, 2)
        if imgType == 0 and validAmt < validNum:
            shutil.move("data/" + str(i) + ".txt", "valid/" + str(i) + ".txt")
            shutil.move("data/" + str(i) + ".jpg", "valid/" + str(i) + ".jpg")
            validAmt += 1
        elif imgType == 1 and testAmt < testNum:
            shutil.move("data/" + str(i) + ".txt", "test/" + str(i) + ".txt")
            shutil.move("data/" + str(i) + ".jpg", "test/" + str(i) + ".jpg")
            testAmt += 1
        else:
            shutil.move("data/" + str(i) + ".txt", "train/" + str(i) + ".txt")
            shutil.move("data/" + str(i) + ".jpg", "train/" + str(i) + ".jpg")
    
    
#slow, should probably be reworked
else:
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