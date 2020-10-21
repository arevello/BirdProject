'''
Created on Jul 17, 2020

@author: Alex
'''

import matplotlib.pyplot as plt
import pickle
import numpy as np
import random

from osgeo import gdal
from osgeo.gdalconst import *

import mathUtils
import birdConstants

import cv2

class ImageUtilities(object):
    '''
    classdocs
    '''
    mu = mathUtils.MathUtilities()
    bc = birdConstants.BirdConstants()
    dumpFile = []
    
    #comment method to debug
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
                            
                            #data2 = bands2.ReadAsArray(xOff-200, yOff-200, 400, 400)
                            #plt.imshow(data2)
                            #plt.show()
                            
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
            
    def getImagesForVIA(self, centers, size, file):
        idx = 0
        for c in centers:
            gdal.AllRegister()
                
            fh = gdal.Open(file, GA_ReadOnly)
            
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
            try:
                xOff, yOff = self.mu.getPixelCoords(c[0], c[1], xOrig, yOrig, pixelWidth, pixelHeight)
            
                data = []
                for i in range(fh.RasterCount):
                    #csvList.append(i)
                    bands2 = fh.GetRasterBand(i+1)
                    #print(bands2.XSize, bands2.YSize)
                    #print(i)

                    data.append(bands2.ReadAsArray(xOff-size/2, yOff-size/2, size, size))
                
                r = data[2]
                g = data[1]
                b = data[0]
                
                #TODO calculate birds in range and what to label them in txt file
                
                filename = 'D:/satImage/viaTest/' + str(idx) + '.jpg'
                
                self.writeJPG(filename, r, g, b, len(r))
                idx += 1
            except Exception as e:
                print(e)
                print("issue with file", file)
                break
    #end debug comment
    
    def assignMeds(self, coords, medI, medV):
        results = []
        idx = 0
        medDists = []
        for c in coords:
            #if not using medoid
            if not c in medV:
                medDists.append(self.mu.closestList(c, medV))
            else:
                medDists.append([medV.index(c), 0])
            idx += 1
        return medDists
    
    def medoidCosts(self, medDists):
        total = 0
        for i in range(len(medDists)):
            total += medDists[i][1]
        return total   
    
    def printMedoids(self, idxs, fileContents, csvIdx):
        coords = []
        for f in range(len(fileContents[csvIdx])):
            x = float(fileContents[csvIdx][f].get('POINT_X'))
            y = float(fileContents[csvIdx][f].get('POINT_Y'))
            coords.append([x,y])
            
        ret = []
        for i in idxs:
            print(coords[i])
            ret.append(coords[i])
        return ret
    
    def kmedoidClustering(self, fileContents, csvIdx):
        random.seed(10)
        clusters = 8
        prevVar = 999999
        var = 0
        
        coords = []
        for f in range(len(fileContents[csvIdx])):
            x = float(fileContents[csvIdx][f].get('POINT_X'))
            y = float(fileContents[csvIdx][f].get('POINT_Y'))
            coords.append([x,y])
        
        while var < prevVar:
            #select c clusters
            medIdxs = []
            medVals = []
            for c in range(clusters):
                medIdxs.append(random.randint(0,len(coords)-1))
                medVals.append(coords[medIdxs[c]])
            
            assignedMeds = self.assignMeds(coords, medIdxs, medVals)
            costCur = self.medoidCosts(assignedMeds)
            
            bestMedIdxs = medIdxs.copy()
            bestMedVals = medVals.copy()
            bestAssignMeds = assignedMeds.copy()
            bestCost = costCur
            
            while 1:
                print(bestMedIdxs)
                tempBestMedIdxs = bestMedIdxs
                tempBestMedVals = bestMedVals
                tempBestAssignMeds = bestAssignMeds
                tempBestCost = bestCost
                
                #for each medoid
                for m in range(len(bestMedIdxs)):
                    for c in range(len(coords)):
                        if bestMedIdxs[m] != c and not c in bestMedIdxs:
                            tempMedIdxs = bestMedIdxs.copy()
                            tempMedVals = bestMedVals.copy()
                            
                            tempMedIdxs[m] = c
                            tempMedVals[m] = coords[c]
                            
                            tempAssignedMeds = self.assignMeds(coords, tempMedIdxs, tempMedVals)
                            tempCostCur = self.medoidCosts(tempAssignedMeds)
                            
                            if tempCostCur < tempBestCost:
                                tempBestMedIdxs = tempMedIdxs
                                tempBestMedVals = tempMedVals
                                tempBestAssignMeds = tempAssignedMeds
                                tempBestCost = tempCostCur
                
                if tempBestCost < bestCost:
                    bestMedIdxs = tempBestMedIdxs
                    bestMedVals = tempBestMedVals
                    bestAssignMeds = tempBestAssignMeds
                    bestCost = tempBestCost
                else:
                    print(bestMedIdxs)
                    exit()
                    break
            
            clusters += 1

    def dumpFile(self, fileName):
        with open(fileName, 'wb') as outfile:
            pickle.dump(self.dumpFile, outfile, pickle.HIGHEST_PROTOCOL)
            
    def openDump(self, fileName):
        with open(fileName, 'rb') as infile:
            loadFile = pickle.load(infile)
            return loadFile
        
    def createJPGs(self, infile):
        for i in range(len(infile)):
            for j in range(len(infile[i])):
                filename = "D:/satImage/jpgs/" + str(infile[i][j][0]) + "_" + str(infile[i][j][1]) + "_" + str(i) + "_" + str(j) + ".jpg"
                r = infile[i][j][2].flatten()
                g = infile[i][j][3].flatten()
                b = infile[i][j][4].flatten()
                
                
                self.writeJPG(filename, r, g, b, 30)
                
    #needs 1d array of rgb of same size
    def writeJPG(self, filename, r, g, b, size):
        result = []
        result.extend(r)
        result.extend(g)
        result.extend(b)
        result = np.array(result)

        result2 = result.reshape(3, size*size).T
        result2 = result2.reshape(size, size, 3)
        
        cv2.imwrite(filename, result2)
        
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
                
                if infile[a][b][0] != 2 and infile[a][b][0] != 0: 
                    infile[a][b].append(mask)
                    print(birdConstants.BirdConstants.specieStr[infile[a][b][0]], infile[a][b][0])
                    plt.imshow(data)
                    plt.show()
    #                 
                    plt.imshow(mask)
                    plt.show()
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
        