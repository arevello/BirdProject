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