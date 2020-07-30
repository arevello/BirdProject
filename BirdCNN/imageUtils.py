'''
Created on Jul 17, 2020

@author: Alex
'''

import matplotlib.pyplot as plt
import pickle
import numpy as np

from osgeo import gdal
from osgeo.gdalconst import *

import mathUtils
import birdConstants

class ImageUtilities(object):
    '''
    classdocs
    '''
    mu = mathUtils.MathUtilities()
    bc = birdConstants.BirdConstants()
    dumpFile = []
    
    def parseImages(self, tifFiles, tifIdx, csvFiles, fileContents):
        gdal.AllRegister()

        #appleIslandFile = 'D:\Appledore\AppledoreIsland_SmuttynoseIsland_CedarIsland_81191_81182_81194_20190531_Ortho_Multi_2CM.tif'
        
        csvIdx = 0
        for t in range(len(tifFiles)):
            
            fh = gdal.Open(tifFiles[t], GA_ReadOnly)
            
            while tifIdx[csvIdx] < t:
                csvIdx += 1
            
            self.dumpFile = []
            
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
                    
                    #format: [tifFile][csvFile][species, behavior, [data]]
                    try:
                        xOff, yOff = self.mu.getPixelCoords(float(fileContents[csvIdx][f].get('POINT_X')), float(fileContents[csvIdx][f].get('POINT_Y')), xOrig, yOrig, pixelWidth, pixelHeight)
                        species = self.bc.strToSpecies(fileContents[csvIdx][f].get('Species'))
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
            self.dumpFile.append(tifList)

    def dumpFile(self, fileName):
        with open(fileName, 'wb') as outfile:
            pickle.dump(self.dumpFile, outfile, pickle.HIGHEST_PROTOCOL)
            
    def openDump(self, fileName):
        with open(fileName, 'rb') as infile:
            loadFile = pickle.load(infile)
            return loadFile
        
    def generateMask(self, infile, closeGap=False):
        print("temp")
        
        #print(infile[0][2])
        
        for a in range(len(infile)):
            for b in range(len(infile[a])):
                
                data = infile[a][b][2]
                mask = np.zeros((30,30))
                
                #start 14,15
                rowCount = 2
                start = 14
                stop = 15
                firstIter = True
                thresh = 100
                while start >= 0 and stop <= 29:
                    for i in range(start, stop+1):
                        for j in range(start, stop+1):
                            if firstIter or data[i][j] >= thresh:
                                mask[i][j] = 1
                    firstIter = False
                    start -= 1
                    stop += 1
                
                infile[a][b].append(mask)
#                 plt.imshow(data)
#                 plt.show()
#                 
#                 plt.imshow(mask)
#                 plt.show()
        #while not at edge and new additions made
        #get surrounding ring of prev center
        #if above thresh and at least 1 mask neighbor set to 1
        #else 0
        
        #for each pixel
        #if has >= thresh positive set to 1
        
        #edge detection????

    def __init__(self):
        '''
        Constructor
        '''
        mu = mathUtils.MathUtilities()
        bc = birdConstants.BirdConstants()
        