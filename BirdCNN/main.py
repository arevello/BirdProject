'''
Created on Jun 8, 2020

@author: Alex
'''

import fileUtils
import imageUtils

fu = fileUtils.FileUtilities()
iu = imageUtils.ImageUtilities()

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

#iu.kmedoidClustering(fileContents, 1)
c = iu.printMedoids([796, 1543, 633, 1250, 1131, 172, 364, 1642], fileContents, 1)
#iu.getImagesForVIA(c, 800, tifFiles[0], "train")
c2 = iu.printMedoids([173, 562, 1312], fileContents, 1)
iu.getImagesForVIA(c2, 800, tifFiles[0], "val")

#iu.parseImages(tifFiles, tifIdx, csvFiles, fileContents)
#iu.dumpFile('D:\\trainImg.pkl')

trainData = iu.openDump('D:\\satImage\\trainImg.pkl')
#iu.createJPGs(trainData)
#iu.generateMask(trainData)
