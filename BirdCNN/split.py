from os import listdir
from os.path import isfile, join
import random
import shutil

darknet = True

birdDict = dict()
trainDict = dict()
testDict = dict()
validDict = dict()
trainDictMax = dict()
testDictMax = dict()
validDictMax = dict()
dirName = "data"

def getBirdsInFile(filename):
    dictTemp = dict()
    fh = open(filename)
    for l in fh:
        try:
            #print(l)
            spc = int(l[0])
        except Exception as e:
            spc = 6
        if not dictTemp.__contains__(spc):
            dictTemp[spc] = 1
        else:
            dictTemp[spc] = dictTemp[spc] + 1
    return dictTemp

def addFileToDict(dictToAddTo, fileDict):
    for k in fileDict.keys():
        if not dictToAddTo.__contains__(k):
            dictToAddTo[k] = fileDict[k]
        else:
            dictToAddTo[k] = dictToAddTo[k] + fileDict[k]
    return dictToAddTo

#get the key with the largest amount of values
def majorityKey(fileDict):
    kIdx = -1
    kVal = -1
    for k in fileDict.keys():
        if fileDict[k] > kVal:
            kIdx = k
    return kIdx

def getOverallIncrease(checkDict, fileDict, fileKey):
    if checkDict == 0:
        return fileDict[fileKey] / trainDictMax[fileKey]
    elif checkDict == 1:
        return fileDict[fileKey] / testDictMax[fileKey]
    else:
        return fileDict[fileKey] / validDictMax[fileKey]

def getOverallPercent(checkDict, fileDict, fileKey):
    if checkDict == 0:
        return (fileDict[fileKey] + trainDict[fileKey]) / trainDictMax[fileKey]
    elif checkDict == 1:
        return (fileDict[fileKey] + testDict[fileKey]) / testDictMax[fileKey]
    else:
        return (fileDict[fileKey] + validDict[fileKey]) / validDictMax[fileKey]

#test each folder with the file, return folder with smallest overall increase
#that is still less than others based on the majority key
def testFoldersWithDict(fileDict):
    fileKey = majorityKey(fileDict)
    trainOverallInc = getOverallIncrease(0, fileDict, fileKey)
    trainOverallPerc = getOverallPercent(0, fileDict, fileKey)

    testOverallInc = getOverallIncrease(1, fileDict, fileKey)
    testOverallPerc = getOverallPercent(1, fileDict, fileKey)

    validOverallInc = getOverallIncrease(2, fileDict, fileKey)
    validOverallPerc = getOverallPercent(2, fileDict, fileKey)

    if testOverallPerc < trainOverallPerc and testOverallPerc < validOverallPerc:
        addFileToDict(testDict, fileDict)
        return 1
    elif validOverallPerc < trainOverallPerc and validOverallPerc < testOverallPerc:
        addFileToDict(validDict, fileDict)
        return 0
    else:
        addFileToDict(trainDict, fileDict)
        return 2

if darknet:
    amtFiles = int(len([f for f in listdir(dirName) if isfile(join(dirName, f))])/2)
    validNum = round(amtFiles * .2)
    testNum = round(amtFiles * .1)
    trainNum = amtFiles - testNum - validNum
    print(validNum, testNum)
    for i in range(amtFiles):
        #print(str(i) + ".txt")
        fileDict = getBirdsInFile(dirName + "/" + str(i) + ".txt")
        addFileToDict(birdDict, fileDict)

    for k in birdDict.keys():
        trainDictMax[k] = round(birdDict[k] * .7)
        validDictMax[k] = round(birdDict[k] * .2)
        testDictMax[k] = round(birdDict[k] * .1)
        trainDict[k] = 0
        validDict[k] = 0
        testDict[k] = 0

    for i in range(amtFiles):
        imgType = testFoldersWithDict(getBirdsInFile(dirName + "/" + str(i) + ".txt"))
        if imgType == 0:
            shutil.move("data/" + str(i) + ".txt", "valid/" + str(i) + ".txt")
            shutil.move("data/" + str(i) + ".jpg", "valid/" + str(i) + ".jpg")
        elif imgType == 1:
            shutil.move("data/" + str(i) + ".txt", "test/" + str(i) + ".txt")
            shutil.move("data/" + str(i) + ".jpg", "test/" + str(i) + ".jpg")
        else:
            shutil.move("data/" + str(i) + ".txt", "train/" + str(i) + ".txt")
            shutil.move("data/" + str(i) + ".jpg", "train/" + str(i) + ".jpg")

    print(birdDict)
    print(trainDictMax)
    print(trainDict)
    print(testDictMax)
    print(testDict)
    print(validDictMax)
    print(validDict)
    exit()
    
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
