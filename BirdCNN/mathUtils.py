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
    
    def pointInBox(self, boxCenter, boxWidth, boxHeight, point):
        if(point[0] >= (boxCenter-(boxWidth/2)) and point[0] <= (boxCenter+(boxWidth/2))):
            if(point[1] >= (boxCenter-(boxHeight/2)) and point[1] <= (boxCenter+(boxHeight/2))):
                return True
        return False