'''
Created on Jun 8, 2020

@author: Alex
'''

class BirdConstants(object):

    #species classifications
    HERG = 0
    GBBG = 1
    COEI = 2
    #COEI_M = 2
    #COEI_F = 3
    TERN = 4
    DCCO = 5
    CANG = 6
    GBHE = 7
    LAGU = 8
    
    #unsure of
    SNEG = 9
    BAEA = 10
    GLIB = 11
    BCNE = 12
    BLGU = 13
    ATPU = 14
    TernSPP = 15
    OTHER = 16
    
    specieStrAll = ["HERG", "GBBG", "COEI", "TERN", "DCCO", "CANG", "GBHE", "LAGU", "SNEG", "BAEA", "GLIB", "BCNE", "BLGU", "ATPU", "Tern spp", "Other"]
    specieStrUseful = ["HERG", "GBBG", "DCCO", "COEI", "Tern spp"]
    
    #numSpeciesClass = 9
    
    #behavior classifications
    roosting = 0
    nesting = 1
    flying = 2
    
    numBehaviorClass = 3
    
#     def strToSpecies(self, spcStr):
#         idx = 0
#         while idx < len(self.specieStrAll):
#             if spcStr == self.specieStrAll[idx]:
#                 return idx
#             idx += 1
#         print("cant find match for ", spcStr)
#         return 16

    def __init__(self):
        '''
        Constructor
        '''
        