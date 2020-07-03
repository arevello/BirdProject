'''
Created on Jun 8, 2020

@author: Alex
'''

class BirdConstants(object):

    #species classifications
    HERG = 0
    GBBG = 1
    COEI_M = 2
    COEI_F = 3
    TERN = 4
    DCCO = 5
    CANG = 6
    GBHE = 7
    LAGU = 8
    
    numSpeciesClass = 9
    
    #behavior classifications
    roosting = 0
    nesting = 1
    flying = 2
    
    numBehaviorClass = 3


    def __init__(self, params):
        '''
        Constructor
        '''
        