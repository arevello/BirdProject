'''
Created on Jun 8, 2020

@author: Alex
'''

import birdConstants
import fileUtils
import mathUtils
import matplotlib.pyplot as plt
import pickle

from osgeo import gdal
from osgeo.gdalconst import *

fu = fileUtils.FileUtilities()
mu = mathUtils.MathUtilities()
bc = birdConstants.BirdConstants()

#fu.removeCsvFiles('D:\\')

#convert to csv
#fu.searchFiles('D:\\', ".xlsx", True)

#get csv files
csvFiles,tifIdx = fu.searchFiles('D:\\', ".csv", pairWithTif=True)
tifFiles = fu.searchFiles('D:\\', ".tif")

print(csvFiles)
print(tifIdx)

#13 x
#14 y
# AppledoreIsland_SmuttynoseIsland_CedarIsland_81191_81182_81194_20190531_Ortho_Multi_2CM.tif

fileContents = []
for i in range(len(csvFiles)):
    fileContents.append(fu.openCsvFile(csvFiles[i]))


gdal.AllRegister()

#appleIslandFile = 'D:\Appledore\AppledoreIsland_SmuttynoseIsland_CedarIsland_81191_81182_81194_20190531_Ortho_Multi_2CM.tif'

csvIdx = 0
for t in range(len(tifFiles)):
    
    fh = gdal.Open(tifFiles[t], GA_ReadOnly)
    
    while tifIdx[csvIdx] < t:
        csvIdx += 1
    
    dumpFile = []
    
    while tifIdx[csvIdx] == t:
        
        tifList = []
        print(t, tifFiles[tifIdx[csvIdx]], csvFiles[csvIdx])
        
        if fh is None:
            print("failed to open")
            exit(1)
        
        #for i in range(len(test)):
            #print(i)
            #print(test[i][13], test[i][14])
            #print(test[i])
        
        cols = fh.RasterXSize
        rows = fh.RasterYSize
        bands = fh.RasterCount
        transform = fh.GetGeoTransform()
        xOrig = transform[0]
        yOrig = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]
        
        for f in range(len(fileContents[csvIdx])):
            csvList = []
            #dumpFile.append([])
            #print(fileContents[csvIdx][f].get('POINT_X'))
            
            try:
                xOff, yOff = mu.getPixelCoords(float(fileContents[csvIdx][f].get('POINT_X')), float(fileContents[csvIdx][f].get('POINT_Y')), xOrig, yOrig, pixelWidth, pixelHeight)
                species = bc.strToSpecies(fileContents[csvIdx][f].get('Species'))
                behavior = int(fileContents[csvIdx][f].get('Behavior'))
                csvList.append(species)
                csvList.append(behavior)
            
                for i in range(fh.RasterCount):
                    #csvList.append(i)
                    bands2 = fh.GetRasterBand(i+1)
                    #print(bands2.XSize, bands2.YSize)
                    #print(i)

                    data = bands2.ReadAsArray(xOff-15, yOff-15, 30, 30)
                    csvList.append(data)
                    #plt.imshow(data)
                    #plt.show()
            except Exception as e:
                print(e)
                print("issue with file", csvFiles[csvIdx])
                break
            
            tifList.append(csvList)
        
        csvIdx += 1
        if csvIdx >= len(tifIdx):
            break
    
    gdal.Unlink(tifFiles[t])
    dumpFile.append(tifList)
    
with open('D:\\trainImg.pkl', 'wb') as outfile:
    pickle.dump(dumpFile, outfile, pickle.HIGHEST_PROTOCOL)