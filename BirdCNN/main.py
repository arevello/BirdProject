'''
Created on Jun 8, 2020

@author: Alex
'''

import random
import fileUtils
import imageUtils
import os

fu = fileUtils.FileUtilities()
iu = imageUtils.ImageUtilities()

trainWidth = 1000
trainHeight = 1000

#fu.removeCsvFiles('D:\\')

#convert to csv
#fu.searchFiles('D:\\', ".xlsx", True)

#get csv files
csvFiles,tifIdx = fu.searchFiles('D:\\satImage\\', ".csv", pairWithTif=True)
tifFiles = fu.searchFiles('D:\\satImage\\', ".tif")

#13 x
#14 y
# AppledoreIsland_SmuttynoseIsland_CedarIsland_81191_81182_81194_20190531_Ortho_Multi_2CM.tif

fileContents = []
for i in range(len(csvFiles)):
    fileContents.append(fu.openCsvFile(csvFiles[i]))

#get width height of files, determine num of clusters based on
#iu.kmedoidClustering(fileContents, 1, 8)

#translate width/height of imgs to coord, determine which birds are in region, keep count of them then print
#c = iu.printMedoids([796, 1543, 633, 1250, 1131, 172, 364, 1642], fileContents, 1)
#iu.getImagesForVIA(c, 800, tifFiles[0], "train")
#c2 = iu.printMedoids([173, 562, 1312], fileContents, 1)
#iu.getImagesForVIA(c2, 256, tifFiles[0], "val")
#iu.getImagesForVIA(c, 256, tifFiles[0], "train")

#print(csvFiles[29])
#exit()

random.seed(0)

fileCount = 0
jsonTrainStr = "{"
jsonTestStr = "{"

for i in range(len(tifFiles)):
    xO, yO, xW, yW = iu.getTifInfo(tifFiles[i])
    centerlist = iu.getCentersWithCounts(i, xO, yO, xW, yW, csvFiles, tifIdx, 10, fileContents, tifFiles[i], trainWidth, trainHeight)
    justCenters = [row[0] for row in centerlist]
    centerCsvIdx = [row[1] for row in centerlist]
    
    amtOfTest = int(len(centerlist) * .1)
    testIdxs = []
    for i in range(amtOfTest):
        testIdxs.append(random.randint(0, len(centerlist) - 1))
    
    #crop images
    c = iu.getMedoids(justCenters, fileContents, centerCsvIdx)    
    badCenters = iu.getImagesForVIA(c, trainWidth, tifFiles[i], "D:/satImage/viaTest4", fileCount, testIdxs)
    idxDif = 0
    for b in range(len(badCenters)):
        print("deleting",badCenters[b])
        centerlist.pop(badCenters[b] - idxDif)
        idxDif += 1
    jsonTrainTempStr,jsonTestTempStr,fileCount = iu.buildViaJsons(centerlist, fileCount, "D:/satImage/viaTest4", fileContents, testIdxs, trainWidth, 30)
    jsonTrainStr += jsonTrainTempStr + ","
    jsonTestStr += jsonTestTempStr + ","

jsonTrainStr = jsonTrainStr[:-1]
jsonTestStr = jsonTestStr[:-1]
    
jsonTrainStr += "}"
jsonTestStr += "}"
trainJsonFile = open("D:/satImage/viaTest4/train/via_region_data.json", "w+")
testJsonFile = open("D:/satImage/viaTest4/val/via_region_data.json", "w+")
trainJsonFile.write(jsonTrainStr)
testJsonFile.write(jsonTestStr)
trainJsonFile.close()
testJsonFile.close()
exit()

iu.parseImages(tifFiles, tifIdx, csvFiles, fileContents)
iu.dumpFile('D:\\trainImg.pkl')

trainData = iu.openDump('D:\\satImage\\trainImg.pkl')
print(trainData)
#iu.createJPGs(trainData)
#iu.generateMask(trainData)
