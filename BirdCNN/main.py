'''
Created on Jun 8, 2020

@author: Alex
'''

# python3 Users\Alex\Documents\GItHub\BirdProject\BirdCNN\main.py
#ssh -L 5919:localhost:5919 arevello@katahdin.acg.maine.edu

import random
import fileUtils
import imageUtils
import os

fu = fileUtils.FileUtilities()
iu = imageUtils.ImageUtilities()

trainWidth = 608
trainHeight = 608

#fu.removeCsvFiles('D:\\')

#process for new folder
#convert to csv
#fu.searchFiles('D:\\051721_Data_Transfer\\', ".xls", True)
#fu.searchFiles('D:\\051721_Data_Transfer\\', ".xlsx", True)
#exit()

#then rest

#get csv files

filec = 0

def processDirectory(direcname, fileCount):
    csvFiles,tifIdx = fu.searchFiles(direcname, ".csv", pairWithTif=True)
    tifFiles = fu.searchFiles(direcname, ".tif")

    # fileCount = 0 #13107
    # csvFiles,tifIdx = fu.searchFiles('D:\\051721_Data_Transfer\\', ".csv", pairWithTif=True)
    # tifFiles = fu.searchFiles('D:\\051721_Data_Transfer\\', ".tif")
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
    
    #check bird counts from files
    # birdDict = dict()
    # for i in range(len(csvFiles)):
    #     for f in range(len(fileContents[i])):
    #         if not birdDict.__contains__(fileContents[i][f].get('Species')):
    #             birdDict[fileContents[i][f].get('Species')] = 1
    #         else:
    #             birdDict[fileContents[i][f].get('Species')] = birdDict[fileContents[i][f].get('Species')] + 1
    # print(birdDict)
    # exit()
    
    #random.seed(0)
    
    jsonTrainStr = "{"
    jsonTestStr = "{"
    highlightSize = 30
    
    useDarknet = True
    
    birdDict = dict()
    birdDictTest = dict()
    for i in range(len(tifFiles)):
        xO, yO, xW, yW = iu.getTifInfo(tifFiles[i])
        centerlist = iu.getCentersWithCounts(i, xO, yO, xW, yW, csvFiles, tifIdx, 0, fileContents, tifFiles[i], trainWidth, trainHeight)
        justCenters = [row[0] for row in centerlist]
        centerCsvIdx = [row[1] for row in centerlist]
        
        testIdxs = []
        if False:
            amtOfTest = int(len(centerlist) * .1)
            for j in range(amtOfTest):
                testIdxs.append(random.randint(0, len(centerlist) - 1))
        
        #crop images
        c = iu.getMedoids(justCenters, fileContents, centerCsvIdx)
        badCenters, birdDict, birdDictTest, fileCount = iu.getImagesForVIA(fileContents, centerlist, c, trainWidth, tifFiles[i], "D:/satImage/viaTest5", fileCount, testIdxs, birdDict, birdDictTest, highlightSize, darknetFiles=useDarknet)
        idxDif = 0
        for b in range(len(badCenters)):
            #print("deleting",badCenters[b])
            centerlist.pop(badCenters[b] - idxDif)
            idxDif += 1
        if useDarknet == False:
            jsonTrainTempStr,jsonTestTempStr,fileCount = iu.buildViaJsons(centerlist, fileCount, "D:/satImage/viaTest5", fileContents, testIdxs, trainWidth, 30)
        if len(centerlist) != 0 and useDarknet == False:
            jsonTrainStr += jsonTrainTempStr + ","
            jsonTestStr += jsonTestTempStr + ","
    
    print(birdDict)
    print("with augments")
    print(birdDictTest)
    if useDarknet == False:
        jsonTrainStr = jsonTrainStr[:-1]
        jsonTestStr = jsonTestStr[:-1]
            
        jsonTrainStr += "}"
        jsonTestStr += "}"
        trainJsonFile = open("D:/satImage/viaTest5/train/via_region_data.json", "w+")
        testJsonFile = open("D:/satImage/viaTest5/val/via_region_data.json", "w+")
        trainJsonFile.write(jsonTrainStr)
        testJsonFile.write(jsonTestStr)
        trainJsonFile.close()
        testJsonFile.close()
    #exit()
    return fileCount
    
    #iu.parseImages(tifFiles, tifIdx, csvFiles, fileContents)
    #iu.dumpFile('D:\\trainImg.pkl')
    
    #trainData = iu.openDump('D:\\satImage\\trainImg.pkl')
    #print(trainData)
#iu.createJPGs(trainData)
#iu.generateMask(trainData)

filec = processDirectory('D:\\satImage\\', filec)
filec = processDirectory('D:\\051721_Data_Transfer\\', filec)