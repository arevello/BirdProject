'''
Created on Jul 17, 2020

@author: Alex
'''

import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import os

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
    
    def getTifInfo(self, filename):
        gdal.AllRegister()
        fh = gdal.Open(filename, GA_ReadOnly)
        transform = fh.GetGeoTransform()
        xOrig = transform[0]
        yOrig = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]
        gdal.Unlink(filename)
        return xOrig, yOrig, pixelWidth, pixelHeight
    
    #returns list of centers of imgs to extract
    #form [[center, csvFile, [other idxs in img]],...]
    #TODO sort bird dict by lowest amnt first, getting lowers first will usually find highers
    def getCentersWithCounts(self, tifImg, xOrig, yOrig, xWidth, yWidth, csvFiles, tifIdx, minCount, fileContents, filename, trainWidth, trainHeight):
        #print(fileContents[csvIdx][f].get('POINT_X'))
        #format: [tifFile][csvFile][species, behavior, [data]]
        centers = list()
        for i in range(len(csvFiles)):
            if tifImg == tifIdx[i]:
                if minCount > 0:
                    birdDict = dict()
                    selBirds = [0] * len(fileContents[i])
                    for f in range(len(fileContents[i])):
                        if not birdDict.__contains__(fileContents[i][f].get('Species')):
                            birdDict[fileContents[i][f].get('Species')] = 1
                        else:
                            birdDict[fileContents[i][f].get('Species')] = birdDict[fileContents[i][f].get('Species')] + 1
                    
                    #change to amt needed
                    for k in birdDict.keys():
                        if birdDict[k] > minCount:
                            birdDict[k] = minCount
                            
                    
                    for k in birdDict.keys():
                        
                        tries = 0
                        while birdDict[k] > 0 and tries < 100:
                            tries = tries + 1
                            chkIdx = random.randint(0, len(fileContents[i]) - 1)
                            if selBirds[chkIdx] == 0 and k == fileContents[i][chkIdx].get('Species'):
                                selBirds,birdDict,centers = self.getFriends(fileContents, i, chkIdx, trainWidth, xWidth, trainHeight, yWidth, selBirds, birdDict, centers)
                                
                    #TODO if no hits just loop through and find first ones if any exist
                    for k in birdDict.keys():
                        print(k, birdDict[k])
                        newadded = 0
                        for r in range(len(fileContents[i])):
                            if birdDict[k] <= 0:
                                break
                            if selBirds[r] == 0 and fileContents[i][r].get('Species') == k:
                                newadded += 1
                                selBirds,birdDict,centers = self.getFriends(fileContents, i, r, trainWidth, xWidth, trainHeight, yWidth, selBirds, birdDict, centers)
                else:
                    selBirds = [0] * len(fileContents[i])
                    birdDict = dict()
                    for r in range(len(fileContents[i])):
                        if selBirds[r] == 0 and birdConstants.BirdConstants.specieStrUseful.__contains__(fileContents[i][r].get('Species')):
                            selBirds,birdDict,centers = self.getFriends(fileContents, i, r, trainWidth, xWidth, trainHeight, yWidth, selBirds, None, centers)
        return centers
    
    #looks for other birds inside an image centered at i and adds them to the centers list
    #TODO make sure cant get repeat friends instead of just repeat centers???
    def getFriends(self,fileContents,csvIdx,chkIdx,trainWidth,xWidth,trainHeight,yWidth,selBirds,birdDict,centers):
        idxFriends = 0
        friends = []
        # check for other points in
        try:
            for f in range(len(fileContents[csvIdx])):
                if self.mu.pointInBoxTif(float(fileContents[csvIdx][chkIdx].get('POINT_X')), float(fileContents[csvIdx][chkIdx].get('POINT_Y')), trainWidth * xWidth, trainHeight * yWidth, float(fileContents[csvIdx][f].get('POINT_X')), float(fileContents[csvIdx][f].get('POINT_Y'))):
                    idxFriends = idxFriends + 1
                    friends.append(f)
        except Exception as e:
            pass
            #print(fileContents[i][chkIdx])
        
        # do for all found points
        for f in friends:
            selBirds[f] = 1
            if birdDict != None:
                birdDict[fileContents[csvIdx][f].get('Species')] = birdDict[fileContents[csvIdx][f].get('Species')] - 1
        temp = list()
        temp.append(chkIdx)
        temp.append(csvIdx)
        temp.append([xWidth,yWidth])
        temp.append(friends)
        centers.append(temp)
        if friends == []:
            print("idiot", csvIdx)
        return selBirds,birdDict,centers
    
    def buildViaJsons(self, centerlist, fileCount, directory, fileContents, testIdxs, imageWidth, boxWidth):
        jsonTestStr = ""
        jsonTrainStr = ""
        for c in range(len(centerlist)):
            
            trainImage = True
            for i in range(len(testIdxs)):
                if testIdxs[i] == c:
                    trainImage = False
            if trainImage:
                myDirectory = directory + "/train"
            else:
                myDirectory = directory + "/val" 
                    
            jsonStr = ""
            filename = str(fileCount) + ".jpg"
            filesize = os.path.getsize(myDirectory + "/" + filename)
            jsonStr += self.quote(filename + str(filesize)) + ":{" + self.jsonPair("filename",filename) + "," + self.jsonIntPair("size", filesize) + "," + self.quote("regions") + ":["
            
            centerX = int((imageWidth/2) - (boxWidth/2))
            centerY = int((imageWidth/2) - (boxWidth/2))
            centerLat = float(fileContents[centerlist[c][1]][centerlist[c][0]].get('POINT_X'))
            centerLon = float(fileContents[centerlist[c][1]][centerlist[c][0]].get('POINT_Y'))

            for f in range(len(centerlist[c][3])):
                jsonStr += "{" + self.quote("shape_attributes") + ":{"
                #TODO test rectangle vs approx mask to polygon
                centerLatF = float(fileContents[centerlist[c][1]][centerlist[c][3][f]].get('POINT_X'))
                centerLonF = float(fileContents[centerlist[c][1]][centerlist[c][3][f]].get('POINT_Y'))
                distX = int((centerLat / centerlist[c][2][0]) - (centerLatF / centerlist[c][2][0]))
                distY = int((centerLon / centerlist[c][2][1]) - (centerLonF / centerlist[c][2][1]))
                fX = centerX - distX
                fY = centerY - distY
                w = 30
                h = 30
                #print(centerLat, centerLatF, distX, fX)
                #print(centerLon, centerLonF, distY, fY)
                birdType = fileContents[centerlist[c][1]][f].get('Species')
                jsonStr += self.jsonPair("name", "rect") + "," + self.jsonIntPair("x", fX) + "," + self.jsonIntPair("y",fY) + "," + self.jsonIntPair("width", w) + "," + self.jsonIntPair("height", h) + "}"
                jsonStr += "," + self.quote("region_attributes") + ":{" + self.jsonPair("BIRD", birdType) + "}},"
                
            
            jsonStr = jsonStr[:-1]
            
            jsonStr += "]," + self.quote("file_attributes") + ":{}},"
                    
            if trainImage:
                jsonTrainStr += jsonStr
            else:
                jsonTestStr += jsonStr
                
            fileCount += 1
        jsonTrainStr = jsonTrainStr[:-1]
        jsonTestStr = jsonTestStr[:-1]
        return jsonTrainStr,jsonTestStr,fileCount
            
    def jsonPair(self,str1,str2):
        return self.quote(str1) + ":" + self.quote(str2)
    
    def jsonIntPair(self,str1,num):
        return self.quote(str1) + ":" + str(num)
    
    def quote(self, str1):
        return "\"" + str1 + "\"" 
    
    #get 30x30 training images
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
            
    def getImagesForVIA(self, fileContents, centers, medoids, size, file, directory, fileIdx, testIdxs, darknetFiles=False):
        badCenters = []
        idx = fileIdx
        print("getting images")
        birdDict = dict()
        for c in range(len(centers)):
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
                xOff, yOff = self.mu.getPixelCoords(medoids[c][0], medoids[c][1], xOrig, yOrig, pixelWidth, pixelHeight)
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
                
                if darknetFiles:
                    centerX = int((size/2) - (30/2))
                    centerY = int((size/2) - (30/2))
                    centerLat = float(fileContents[centers[c][1]][centers[c][0]].get('POINT_X'))
                    centerLon = float(fileContents[centers[c][1]][centers[c][0]].get('POINT_Y'))
                    ret = ""
                    for f in range(len(centers[c][3])):
                        
                        #TODO test rectangle vs approx mask to polygon
                        centerLatF = float(fileContents[centers[c][1]][centers[c][3][f]].get('POINT_X'))
                        centerLonF = float(fileContents[centers[c][1]][centers[c][3][f]].get('POINT_Y'))
                        distX = int((centerLat / centers[c][2][0]) - (centerLatF / centers[c][2][0]))
                        distY = int((centerLon / centers[c][2][1]) - (centerLonF / centers[c][2][1]))
                        fX = centerX - distX
                        fY = centerY - distY
                        w = 30
                        h = 30
                        #print(centerLat, centerLatF, distX, fX)
                        #print(centerLon, centerLonF, distY, fY)
                        birdType = fileContents[centers[c][3][f]].get('Species')
                        
                        _x      = (fX+30/2) / size # relative position of center x of rect
                        _y      = (fY+30/2) / size # relative position of center y of rect
                        _width  = 30 / size
                        _height = 30 / size
                        
                        ret += str([birdType.lower() for item in birdConstants.BirdConstants.specieStrUseful].index(birdType.lower())) + " " + str(fileContents[centers[c][1]][f].get('Species')) + " " + str(_x) + " " + str(_y) + " " + str(_width) + " " + str(_height) +"\n"
                        if not birdDict.__contains__(birdType):
                            birdDict[birdType] = 1
                        else:
                            birdDict[birdType] = birdDict[birdType] + 1
                        
                    jpgFilename = directory + "/data/" + str(idx) + '.jpg'
                    txtFilename = directory + "/data/" + str(idx) + '.txt'
                    fo = open(txtFilename, "w")
                    fo.write(ret)
                    fo.close()
                else:
                    trainImage = True
                    for i in range(len(testIdxs)):
                        if testIdxs[i] == c:
                            trainImage = False
                    if trainImage:
                        jpgFilename = directory + "/train/" + str(idx) + '.jpg'
                    else:
                        jpgFilename = directory + "/val/" + str(idx) + '.jpg'
                        
                self.writeJPG(jpgFilename, r, g, b, len(r))
                idx += 1
            except Exception as e:
                print(e)
                #print("issue with file", file)
                badCenters.append(c)
                #print(xOff, yOff, c[0], c[1], xOrig, yOrig, pixelWidth, pixelHeight)
        print(file, birdDict)
        return badCenters
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
    
    def getMedoids(self, idxs, fileContents, csvIdx):            
        ret = []
        for i in range(len(idxs)):
            x = float(fileContents[csvIdx[i]][idxs[i]].get('POINT_X'))
            y = float(fileContents[csvIdx[i]][idxs[i]].get('POINT_Y'))
            ret.append([x,y])
        return ret
    
    def kmedoidClustering(self, fileContents, csvIdx, clusters):
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
                    print(birdConstants.BirdConstants.specieStrAll[infile[a][b][0]], infile[a][b][0])
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
        
