
#get dimensions
#figure out hidden layers
#figure out filters and size
#dot prod filter and size of in orig
#for width of og
#  for height of og
#with result 

#use more specific filters in later layers

import mnist_loader
import numpy as np
from math import sqrt
#from keras.datasets import mnist

np.random.seed(0)
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#filters = [[[-1, -1, -1],[0,0,0],[1,1,1]],[[-1, -1, -1],[0,0,0],[1,1,1]],[[-1, -1, -1],[0,0,0],[1,1,1]],[[-1, -1, -1],[0,0,0],[1,1,1]]]
filters = np.array([-1,-1,-1,0,0,0,1,1,1,
                    1,1,1,0,0,0,-1,-1,-1,
                    1,0,-1,1,0,-1,1,0,-1])

width = 28
height = 28
colorDepth = 1
padSize = 1 #add padded row and col?
strideSize = 1 #number of pixels to move over after conv of one block
learningRate = 1e-7

#tested
def shapeFilters(dim, filts):
    ret = []
    lenghtOfFilt = dim*dim
    for i in range(int(len(filts)/lenghtOfFilt)):
        ret.append(np.reshape(filts[i*lenghtOfFilt:(i+1)*lenghtOfFilt],(dim,dim)))
    return ret

filters = shapeFilters(3, filters)
print(filters)

def outputSize(inputW, filterW, padding, stride):
    return (inputW - filterW + 2*padding)/(stride + 1)

def relu(val, derivative=False):
    '''if derivative:
        return 1/(1 + np.exp(-1*val))
    return (np.exp(val))/(1+ np.exp(val))'''
    '''if derivative:
        for i in range(len(val)):
            for j in range(len(val[0])):
                val[i][j] = 1e-5
        return val
    for i in range(len(val)):
        for j in range(len(val[0])):
            if val[i][j] < 0.:
                val[i][j] = 0.
    return val'''
    if derivative:
        return 0
    return np.maximum(val,0)

#assume square
#tested
def reshapeImageWithPadding(image, paddingSize):
    shape = int(sqrt(len(image)))
    stop = shape+paddingSize*2
    print(shape,stop)
    ret = []
    retRow = np.zeros((stop,1))
    if paddingSize != 0:
        for i in range(stop):
            if i==0 or i==(stop-1):
                ret.append(retRow.copy())
            else:
                temp = np.zeros((stop,1))
                temp[1:stop-1] = image[shape*(i-1):shape*i].copy()
                #ret.append(image[shape*(i-1):shape*i].copy())
                #ret.append(0)
                ret.append(temp)
    else:
        ret.append(image.copy())
            
    return np.reshape(ret, (stop,stop))

#tested
def getSubsetOfImage(image, size, x, y):
    ret = []
    for i in range(size):
        ret.append(np.array(image[x+i][y:y+size]))
    return np.array(ret)

#tested
def avgPoolFilters(input, poolSize):
    ret = []
    for f in range(len(input)):
        d,n = input[f].shape
        filterRet = []
        xItr = int(d/poolSize)
        yItr = int(n/poolSize)
        for i in range(xItr):
            for j in range(yItr):
                temp = getSubsetOfImage(input[f], poolSize, i*poolSize, j*poolSize)
                d2,n2 = temp.shape
                tempVal = np.sum(temp)/(d2*n2)
                filterRet.append(tempVal)
        filterRet = np.reshape(filterRet, (xItr,yItr))
        ret.append(filterRet)
    return ret

#tested
def maxPoolFilters(input, poolSize):
    ret = []
    for f in range(len(input)):
        d,n = input[f].shape
        filterRet = []
        xItr = int(d/poolSize)
        yItr = int(n/poolSize)
        for i in range(xItr):
            for j in range(yItr):
                temp = getSubsetOfImage(input[f], poolSize, i*poolSize, j*poolSize)
                tempVal = np.amax(temp)
                filterRet.append(tempVal)
        filterRet = np.reshape(filterRet, (xItr,yItr))
        ret.append(filterRet)
    return ret

#tested
def flattenImage(input):
    ret = []
    finalDim = 0
    for f in range(len(input)):
        d,n = input[f].shape
        finalDim += d*n
    return np.reshape(input.copy(), (finalDim,1))

#tested
def createNeuralNetwork(firstLayer, numHiddenLayers, hiddenLayerSizes, outputLayerSize):
    weights = []
    bias = []
    weights.append(np.random.randn(firstLayer,hiddenLayerSizes[0]))
    bias.append(np.zeros((1,hiddenLayerSizes[0])))
    for i in range(0,len(hiddenLayerSizes)):
        weights.append(np.random.randn(hiddenLayerSizes[i],hiddenLayerSizes[i+1]))
        bias.append(np.zeros((1,hiddenLayerSizes[i+1])))
    weights.append(np.random.randn(hiddenLayerSizes[len(hiddenLayerSizes)-1], outputLayerSize))
    bias.append(np.zeros((1, outputLayerSize)))
    return weights, bias
    
#tested
def trainNeuralNetwork(epochs, weights, biases, trainX, trainY):
    for i in range(epochs):
        for j in range(len(trainX)):
            '''
            #might need some variation
            tempX = trainX[j]
            tempX=np.zeros((1,2))
            tempX[0][0] = trainX[j][0]
            tempX[0][1] = trainX[j][1]
            '''
            tempX = trainX[j]
            tempX = np.reshape(tempX, (1,len(tempX)))
            zTemp,activations = feedForward(tempX, weights, biases) 
            weights,biases = backPropogation(tempX, trainY[j], weights, biases, activations, zTemp)
    return weights, biases
 
#tested           
def feedForward(trainData, weights, biases):
    zTemp = []
    activations = []
    zTemp.append(np.dot(trainData, weights[0]) + biases[0])
    activations.append(relu(zTemp[0]))
    for i in range(1,len(weights)):
        zTemp.append(np.dot(activations[i-1], weights[i]) + biases[i])
        if i != len(weights)-1:
            activations.append(relu(zTemp[i]))
    zTemp = zTemp[::-1]
    return zTemp,activations

#tested
def backPropogation(trainData, trainLabel, weights, biases, activations, zTemp):
    loss = np.mean((zTemp[0] - trainLabel)**2)
    
    errors=[]
    deltas=[]
    
    errors.append((zTemp[0] - trainLabel)/float(trainLabel.shape[0]))#1
    errors.append(errors[0].dot(weights[len(weights)-1].T))#2
    
    zTempItr = 1
    for i in range(len(weights) - 2, -1, -1):
        deltas.append(errors[len(errors)-1]*relu(zTemp[zTempItr],derivative=True))#3
        if i != 0:
            errors.append(deltas[len(deltas)-1].dot(weights[i].T))#4
        zTempItr += 1
        
    weights[0] -= learningRate*np.dot(trainData.T,deltas[len(deltas)-1])
    biases[0] -= learningRate*np.sum(deltas[len(deltas)-1],axis = 0)
    
    activationsIter = 0
    deltasIter = len(deltas) - 2
    for i in range(1,len(weights)-1):
        weights[i] -= learningRate*activations[activationsIter].T.dot(deltas[deltasIter])
        biases[i] -= learningRate*np.sum(deltas[deltasIter],axis = 0)
        activationsIter += 1
        deltasIter -= 1
    weights[len(weights)-1] -= learningRate*activations[activationsIter].T*(errors[0])
    biases[len(biases)-1] -= learningRate*np.sum(errors[0])
    return weights,biases

'''poolTest = [[1,1,2,2,1,1,2,2,3,3,4,4,3,3,4,4],[1,1,2,2,3,3,1,1,2,2,3,3]]
poolTest2 = []
poolTest2.append(np.reshape(poolTest[0], (4,4)))
poolTest2.append(np.reshape(poolTest[1], (2,6)))
print(poolTest2)
print(maxPoolFilters(poolTest2,2))'''
'''poolTest = [[1,2,1,2,3,4,3,4,1,2,1,2,3,4,3,4],[1,2,1,2,3,4,3,4]]
poolTest2 = []
poolTest2.append(np.reshape(poolTest[0], (4,4)))
poolTest2.append(np.reshape(poolTest[1], (2,4)))
print(maxPoolFilters(poolTest2,2))
exit()'''

test1 = reshapeImageWithPadding(test_data[0][0], 0)
test2 = reshapeImageWithPadding(test_data[0][0], 1)

'''print(filters[2])
for x in range(2):
    for y in range(2):
        print(getSubsetOfImage(filters[2], 2, x, y))'''

#test_data[0][0-783] is vals [0-1]
#test_data[0][0-9] is label 0 or 1

#one conv layer, one image, one color layer
#for all images
convStep1 = []
step = 3
for i in range(len(filters)):
    convFilterStep = []
    for x in range(10):
        for y in range(10):
            temp = np.sum(getSubsetOfImage(test2, 3, x*step, y*step) * filters[i]) #+ bias
            convFilterStep.append(temp)
    convFilterStep = np.reshape(np.array(convFilterStep), (10,10))
    convStep1.append(convFilterStep)

#pooling
poolStep = maxPoolFilters(convStep1, 2)
flattenStep = flattenImage(poolStep)