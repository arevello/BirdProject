'''
Created on Jun 29, 2020

@author: Alex
'''

class MathUtilities(object):


    def __init__(self):
        '''
        Constructor
        '''
    
    def getPixelCoords(self, xPt, yPt, xOrigin, yOrigin, pixelWidth, pixelHeight):
#         xOff = abs(int(abs((float(xPt) - xOrigin)) / pixelWidth))
#         yOff = abs(int(abs((float(yPt) - yOrigin)) / pixelHeight))
        xOff = int((float(xPt) - xOrigin) / pixelWidth)
        yOff = int((float(yPt) - yOrigin) / pixelHeight)
        
        return xOff, yOff
    
    def coordDistSqrd(self, pt1, pt2):
        return (pt2[1] - pt1[1])**2 + (pt2[0] - pt1[0])**2 
    
    def closestList(self, pt, comparePts):
        idx = 0
        dist = self.coordDistSqrd(pt, comparePts[0])
        for c in range(1, len(comparePts)-1):
            tempDist = self.coordDistSqrd(pt, comparePts[c])
            if(tempDist < dist):
                dist = tempDist
                idx = c
        
        return idx, dist
    
    def pointInBoxTif(self, centerX, centerY, boxWidth, boxHeight, pointX, pointY):
        #print(centerX, centerY, boxWidth, boxHeight, pointX, pointY)
        if(pointX >= (centerX-(boxWidth/2)) and pointX <= (centerX+(boxWidth/2))):
            if(pointY >= (centerY+(boxHeight/2)) and pointY <= (centerY-(boxHeight/2))):
                return True
        return False