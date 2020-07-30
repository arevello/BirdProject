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
csvFiles,tifIdx = fu.searchFiles('D:\\', ".csv", pairWithTif=True)
tifFiles = fu.searchFiles('D:\\', ".tif")

#13 x
#14 y
# AppledoreIsland_SmuttynoseIsland_CedarIsland_81191_81182_81194_20190531_Ortho_Multi_2CM.tif

fileContents = []
for i in range(len(csvFiles)):
    fileContents.append(fu.openCsvFile(csvFiles[i]))

iu.parseImages(tifFiles, tifIdx, csvFiles, fileContents)
#iu.dumpFile('D:\\trainImg.pkl')

trainData = iu.openDump('D:\\trainImg.pkl')
iu.generateMask(trainData)

print(trainData[0][0])
