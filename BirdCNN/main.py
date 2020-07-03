'''
Created on Jun 8, 2020

@author: Alex
'''

import birdConstants
import fileUtils
import mathUtils
import matplotlib.pyplot as plt

from osgeo import gdal
from osgeo.gdalconst import *

fu = fileUtils.FileUtilities()
mu = mathUtils.MathUtilities()

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

appleIslandFile = 'D:\Appledore\AppledoreIsland_SmuttynoseIsland_CedarIsland_81191_81182_81194_20190531_Ortho_Multi_2CM.tif'

csvIdx = 0
for t in range(len(tifFiles)):
    
    fh = gdal.Open(tifFiles[t], GA_ReadOnly)
    
    while tifIdx[csvIdx] < t:
        csvIdx += 1
    
    while tifIdx[csvIdx] == t:
        
        print(t, tifIdx[csvIdx], csvIdx)
        
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
        
        print(fileContents[csvIdx][0].get('X_POINT'))
        '''for idx in range(1, 11):
            
            xOff, yOff = mu.getPixelCoords(float(fileContents[csvIdx][idx].get('X_POINT')), float(fileContents[csvIdx][idx].get('Y_POINT')), xOrig, yOrig, pixelWidth, pixelHeight)
            
            for i in range(fh.RasterCount):
                bands2 = fh.GetRasterBand(i+1)
                #print(bands2.XSize, bands2.YSize)
                #print(i)
                
                data = bands2.ReadAsArray(xOff-15, yOff-15, 30, 30)
                #plt.imshow(data)
                #plt.show()'''
        
        csvIdx += 1
    
    gdal.Unlink(tifFiles[t])