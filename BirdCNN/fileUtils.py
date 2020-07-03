'''
Created on Jun 8, 2020

@author: Alex
'''

import xlrd
import csv
import os

class FileUtilities():
    '''
    classdocs
    '''
    
    def openCsvFile(self, filename):
        fh = open(filename, "r")
        reader = csv.DictReader(fh)
        contents = []
        for line in reader:
            contents.append(line)
        
        '''idx = 0
        line = fh.readline().strip()
        tokens = line.split(",")
        contents.append(tokens)
        for line in fh:
            idx += 1
            line = line.strip()
            tokens = line.split(",")
            contents.append(tokens)'''
            
        fh.close()
        return contents

    def searchFiles(self, directory='.', extension='', createCsv=False, pairWithTif=False):
        foundFiles = []
        tifIndexes = []
        extension = extension.lower()
        tifIdx = -1
        tifFile = 0
        for dirpath, dirnames, files in os.walk(directory):
            for name in files:
                if extension and name.lower().endswith(extension):
                    if createCsv:
                        fileSplit = name.split(".")
                        csvName = fileSplit[0] + ".csv"
                        if csvName not in files:
                            print("creating " + csvName)
                            self.csvFromExcel(dirpath, name, csvName)
                        else:
                            print("not creating")
                    foundFiles.append(os.path.join(dirpath, name))
                    if pairWithTif:
                        tifIndexes.append(tifIdx)
                    
                elif pairWithTif:
                    if extension and name.lower().endswith(".tif"):
                        if not tifFile == name.lower():
                            tifFile = name.lower()
                            tifIdx += 1 
        if pairWithTif:
            return foundFiles, tifIndexes
        return foundFiles
    
    #EXTREMELY DANGEROUS
    #ONLY USE ON FILES YOU CREATED SO YOU DON'T NEED TO ASK FOR COPIES LIKE A FUCKING MORON
    def removeCsvFiles(self, directory):
        foundFiles = []
        extension = ".csv"
        extension = extension.lower()
        for dirpath, dirnames, files in os.walk(directory):
            for name in files:
                if extension and name.lower().endswith(extension):
                    os.remove(os.path.join(dirpath, name))
    
    def csvFromExcel(self, directory, wbName, csvName):
        wb = xlrd.open_workbook(os.path.join(directory, wbName))
        sh = wb.sheet_by_index(0)
        csvFile = open(os.path.join(directory, csvName), 'w')
        #wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)
    
        for rownum in range(sh.nrows):
            temp = sh.row_values(rownum)
            line = ""
            for t in range(len(temp)):
                tempToken = str(temp[t])
                tempToken = tempToken.replace(",", ";")
                line += tempToken
                if not t == len(temp) - 1:
                    line += ","
            csvFile.write(line + "\n")
    
        csvFile.close()


    def __init__(self):
        '''
        Constructor
        '''
        