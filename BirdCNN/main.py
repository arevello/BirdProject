
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
import datetime
import matplotlib.pyplot as plt
import pickle
#from keras.datasets import mnist

np.random.seed(0)

timeStart = datetime.datetime.now()

trainSize = 1000
testSize = 100
hiddenLay = [100]
'''convs = (10, 8)
convstep = (3,1)
pool = 1'''
convs = (28, 12)
convstep = (1,1)
pool = 2
learningRate = 1e-7
epochs = 10000

training_data_full, validation_data_full, test_data_full = mnist_loader.load_data_wrapper()
#print(len(training_data), len(test_data))

training_data_temp = training_data_full[:trainSize]
test_data_temp = training_data_full[:trainSize]
real_test_data = test_data_full[:testSize]

#filters = [[[-1, -1, -1],[0,0,0],[1,1,1]],[[-1, -1, -1],[0,0,0],[1,1,1]],[[-1, -1, -1],[0,0,0],[1,1,1]],[[-1, -1, -1],[0,0,0],[1,1,1]]]
'''filters = np.array([-1,-1,-1,0,0,0,1,1,1,
                    1,1,1,0,0,0,-1,-1,-1,
                    1,0,-1,1,0,-1,1,0,-1,
                    -1,0,1,-1,0,1,-1,0,1,
                    -1,-1,0,-1,0,1,0,1,1,
                    0,-1,-1,1,0,-1,1,1,0,
                    1,1,0,1,0,-1,0,-1,-1,
                    0,1,1,-1,0,1,-1,-1,0])'''
filters = np.array([-1,-1,-1,0,0,0,1,1,1,
                    1,1,1,0,0,0,-1,-1,-1,
                    1,0,-1,1,0,-1,1,0,-1,
                    -1,0,1,-1,0,1,-1,0,1])

width = 28
height = 28
colorDepth = 1
padSize = 1 #add padded row and col?
strideSize = 1 #number of pixels to move over after conv of one block

def buildTestImage(size):
    #xCoords = [2,5,4,3,7,8,6,1,9,0]
    #yCoords = [5,7,3,1,8,0,4,2,6,9]
    image = np.zeros((size*size,1))
    imageSeg = np.zeros((size*size,1))
    
    #for i in range(len(xCoords)):
    for h in range(10):
        for i in range(10):
            for j in range(28):
                #image[xCoords[i]*28+j*size+yCoords[i]*size*28:xCoords[i]*28+j*size+yCoords[i]*size*28+28] = training_data_temp[i][0][j*28:j*28+28]
                image[h*28+j*size+i*size*28:h*28+j*size+i*size*28+28] = training_data_temp[h*10+i][0][j*28:j*28+28]
                val = 0
                for k in range(10):
                    if training_data_temp[h*10+i][1][k] == 1:
                        val = k
                for k in range(28):
                    imageSeg[h*28+j*size+i*size*28+k] = val
    
    imageSeg = np.reshape(imageSeg, (size, size))
    
    plt.imshow(imageSeg)
    plt.show()
    image = np.reshape(image, (size,size))
    
    return image, imageSeg

#tested
def shapeFilters(dim, filts):
    ret = []
    lenghtOfFilt = dim*dim
    for i in range(int(len(filts)/lenghtOfFilt)):
        ret.append(np.reshape(filts[i*lenghtOfFilt:(i+1)*lenghtOfFilt],(dim,dim)))
    return ret

filters = shapeFilters(3, filters)

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

def padAlreadySquare(image, paddingSize):
    n,d = image.shape
    ret = []
    for i in range(paddingSize):
        ret.append(np.zeros((paddingSize*2+n)))
    for i in range(n):
        temp = np.zeros((paddingSize*2+n))
        test2 = [image[i].copy()]
        temp[paddingSize:paddingSize+n] = image[i][:].copy()
        ret.append(temp)
    for i in range(paddingSize):
        ret.append(np.zeros((paddingSize*2+n)))
    return np.reshape(ret, (paddingSize*2+n, paddingSize*2+n))

#assume square, input line
#tested only with 1 TODO dynamic
def reshapeImageWithPadding(image, paddingSize):
    shape = int(sqrt(len(image)))
    stop = shape+paddingSize*2
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
    for i in range(0,len(hiddenLayerSizes)-1):
        weights.append(np.random.randn(hiddenLayerSizes[i],hiddenLayerSizes[i+1]))
        bias.append(np.zeros((1,hiddenLayerSizes[i+1])))
    weights.append(np.random.randn(hiddenLayerSizes[len(hiddenLayerSizes)-1], outputLayerSize))
    bias.append(np.zeros((1, outputLayerSize)))
    return weights, bias

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=1, keepdims=True)
    
#tested
def trainNeuralNetwork(epochs, weights, biases, trainX, trainY):
    for i in range(epochs):
        if i%100 == 0:
            print(int(i/epochs*100), " % done")
        for j in range(len(trainX)):
            tempX = trainX[j]
            tempX = np.reshape(tempX, (1,len(tempX)))
            zTemp,activations = feedForward(tempX, weights, biases) 
            label = trainY[j][1]
            weights,biases = backPropogation(tempX, label.T, weights, biases, activations, zTemp)
            #weights,biases = trainOneLayerTest(tempX, weights, biases, trainY[j][1].T)
    return weights, biases
 
#tested           
#zTemp[0] = y hat
def feedForward(trainData, weights, biases):
    zTemp = []
    activations = []
    test = np.dot(trainData, weights[0])
    zTemp.append(test + biases[0])
    activations.append(relu(zTemp[0]))
    for i in range(1,len(weights)):
        #old way
        '''zTemp.append(np.dot(activations[i-1], weights[i]) + biases[i])
        if i != len(weights)-1:
            activations.append(relu(zTemp[i]))'''
        #new way
        if i != len(weights)-1:
            zTemp.append(np.dot(activations[i-1], weights[i]) + biases[i])
            activations.append(relu(zTemp[i]))
        else:
            zTemp.append(np.dot(activations[i-1], weights[i]) + biases[i])
            activations.append(softmax(zTemp[i]))
    zTemp = zTemp[::-1]
    return zTemp,activations

def trainOneLayerTest(trainData, weights, biases, trainLabel):
    zh = np.dot(trainData, weights[0]) + biases[0]
    ah = relu(zh)

    # Phase 2
    zo = np.dot(ah, weights[1]) + biases[1]
    ao = softmax(zo)

########## Back Propagation

########## Phase 1

    dcost_dzo = ao - trainLabel
    dzo_dwo = ah

    dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

########## Phases 2

    dzo_dah = weights[1]
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = relu(zh, derivative=True)
    dzh_dwh = trainData
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    # Update Weights ================

    weights[0] -= learningRate * dcost_wh
    biases[0] -= learningRate * dcost_bh.sum(axis=0)

    weights[1] -= learningRate * dcost_wo
    biases[1] -= learningRate * dcost_bo.sum(axis=0)
    return weights, biases

#tested
def backPropogation(trainData, trainLabel, weights, biases, activations, zTemp):
    
    errors=[]
    deltas=[]
    '''
    #new fucked way
    dZ = zTemp[0] - trainLabel
    dW = dZ.dot(activations[len(activations)-1].T)
    dB = np.sum(dZ, axis = 1, keepdims=True)
    daPrev = dZ.dot(weights[len(weights) - 1].T)
    
    for i in range(len(weights)-1, 0, -1):
        dZ = daPrev *relu(zTemp[i], derivative=True)
        dW = dZ.dot(activations[i-1].T)
        dB = np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
            daPrev = weights[i].T.dot(dZ.T)
            
        errors.append(dW)
        deltas.append(dB)
        
    for i in range(len(errors)):
        weights[i] -= learningRate*dW[i]
        biases[i] -= learningRate*dB[i]
    
    return weights,biases'''
    
    #og way
    #loss = zTemp[0] - trainLabel
    errors.append((zTemp[0] - trainLabel))#/float(trainLabel.shape[0]))#1
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

def calculateAccuracy(output, label):
    labelAns = 0
    for i in range(len(label)):
        if int(label[i]) == 1:
            labelAns = i
    
    guessAns = 0
    guessAnsItr = 0
    for i in range(len(output)):
        if output[i] > guessAns:
            guessAns = output[i]
            guessAnsItr = i
            
    return labelAns, guessAnsItr, guessAns

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

def convolveImageSet(set, padding, newSize, step, poolSize, flatten=True, returnAll=False, alreadySquare=False):
    
    tempPadding = []
    flattenStep = []
    #add padding to all training images
    for i in range(len(set)):
        if alreadySquare:
            tempPadding.append(padAlreadySquare(set[i][0], padding))
        else:
            tempPadding.append(reshapeImageWithPadding(set[i][0], padding))
    
    #for all images
    for i in range(len(tempPadding)):
        #for amount of convs to do
        prevConvStep = []
        prevConvStep.append(tempPadding[i])
        for s in range(len(step)):
        #test_data[0][0][0-783] is vals [0-1]
        #test_data[0][1][0-9] is label 0 or 1
        
            convStep = []
            #for all filters
            for p in range(len(prevConvStep)):
                for f in range(len(filters)):
                    convFilterStep = []
                    for x in range(newSize[s]):
                        for y in range(newSize[s]):
                            temp = np.sum(getSubsetOfImage(prevConvStep[p], 3, x*step[s], y*step[s]) * filters[f]) #+ bias #replace 3 with filter len
                            convFilterStep.append(temp)
                    convFilterStep = np.reshape(np.array(convFilterStep), (newSize[s],newSize[s]))
                    convStep.append(convFilterStep)
                    
            
            '''if i == 0:
                plt.imshow(convStep[0])
                plt.show()'''
            prevConvStep = convStep.copy()
            #pooling
            if(poolSize != 1):
                prevConvStep = maxPoolFilters(prevConvStep, poolSize)
                '''if i == 0:
                    plt.imshow(prevConvStep[0])
                    plt.show()'''
            if returnAll:
                if flatten:
                    flattenStep.append(flattenImage(prevConvStep))
                else:
                    flattenStep.append(prevConvStep)
        if not returnAll:
            if flatten:
                flattenStep.append(flattenImage(prevConvStep))
            else:
                flattenStep.append(prevConvStep)
    return flattenStep

def convolveOnlyPooling(image, iters):
    ret = []
    ret.append(image)
    for i in range(iters):
        ret.append(maxPoolFilters([ret[len(ret)-1]], 2)[0])
        
    return ret

def maxUpsample(image, size, amount):
    if amount == 0:
        return image
    image = maxUpsample(image, size, amount-1)
    ret = np.zeros((len(image)*size, len(image)*size))
    for i in range(len(image)):
        for j in range(len(image[i])):
            for y in range(size):
                for z in range(size):
                    ret[i*size+y][j*size+z] = image[i][j]
    return ret

def upsamplePredictions(preds, size):
    ret = np.zeros((len(preds)*size, len(preds)*size, 10))
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            for y in range(size):
                for z in range(size):
                    ret[i*size+y][j*size+z] = preds[i][j]
    return ret

def predToValue(preds):
    guessAns = 0
    guessAnsItr = 0
    predsTemp = preds
    for i in range(len(predsTemp)):
        if predsTemp[i] > guessAns:
            guessAns = predsTemp[i]
            guessAnsItr = i
    return guessAnsItr

def printAccuracyOfSegmentation(result, test):
    d,n = result.shape
    print(d,n)
    correct = 0
    for i in range(d):
        for j in range(n):
            if result[i][j] == test[i][j]:
                correct += 1
    print("accuracy: ", str(correct/(d*n)))

def fullyConvSemSeg():
    #288x288
    #get 144x144x4, 72x72x16, 36x36x64, 18x18x256, 9x9x1024
    testImageSize = 288
    segImage,classifiedImage = buildTestImage(testImageSize)
    #test = convolveImageSet([[segImage]], 1, (280, 138), (1,1), 2, flatten=False, returnAll=True)
    test = convolveOnlyPooling(segImage, 5)
    #need 2 iters if doing filters
    startRange = 1
    stopRange = 3
    images = []
    preds = dict()
    predItr = 0
    for i in range(len(test)):
        temp = padAlreadySquare(test[i], 14)
        
        #snag all 28x28 squares and feed forward to get classification
                #print("fuck")
        if i >= startRange and i <= stopRange:
            ogN, ogD = test[i].shape
            thisSet = []
            print(temp.shape, test[i].shape)
            for n in range(ogN):
                for d in range(ogD):
                    test2 = getSubsetOfImage(temp, 28, n, d)
                    thisSet.append([test2])
                    
            flattenedStep = convolveImageSet(thisSet, 1, convs, convstep, pool, alreadySquare=True)
            guessImage = []
            guessPreds = []
            for i in range(len(flattenedStep)):
                z,a = feedForward(flattenedStep[i].T, w, b)
                guessAns = 0
                guessAnsItr = 0
                predsTemp = a[len(a)-1].T
                guessPreds.append(predsTemp)
                for i in range(len(predsTemp)):
                    if predsTemp[i] > guessAns:
                        guessAns = predsTemp[i]
                        guessAnsItr = i
                guessImage.append(guessAnsItr)
            
            guessImage = np.reshape(guessImage, (ogN, ogD))
            guessPreds = np.reshape(guessPreds, (ogN, ogD, 10))
            images.append(guessImage)
            predStr = str(ogN) + "," + str(ogD)
            preds[predStr] = guessPreds
            predItr += 1
        
    #deconvolve by max unpooling
    images = images[::-1]
    scaleSize = stopRange
    prevUpsampledVals = []
    for i in range(len(images)):
    
        tempSize = testImageSize
        for s in range(scaleSize):
            tempSize /= 2
        tempSize = int(tempSize)
        sizeStr = str(str(tempSize) + "," + str(tempSize))
        
        if scaleSize == stopRange:
            fullyUpsampledImage = maxUpsample(images[i], 2, scaleSize) #FCN 2**scaleSize
            printAccuracyOfSegmentation(fullyUpsampledImage, classifiedImage)
            plt.imshow(fullyUpsampledImage)
            plt.show()
            print("fuck")
        else:
            #get this ones preds and add to prev
            newPredVals = prevUpsampledVals + preds[sizeStr]

            #get class image from preds
            tempImage = []
            for n1 in range(len(newPredVals)):
                for n2 in range(len(newPredVals[n1])):
                    tempImage.append(predToValue(newPredVals[n1][n2]))
            tempImage = np.resize(tempImage, (tempSize, tempSize))
            #upscale            
            fullyUpsampledImage = maxUpsample(tempImage, 2, scaleSize) #FCN 2**scaleSize
            printAccuracyOfSegmentation(fullyUpsampledImage, classifiedImage)
            plt.imshow(fullyUpsampledImage)
            plt.show()
            print("fuck")
        
        if scaleSize != startRange:
            #at the end so no reason to upscale predictions again
            prevUpsampledVals = upsamplePredictions(preds[sizeStr], 2)
            
        scaleSize -= 1
    
    '''for i in range(startRange, stopRange):
        for i in range
        upsamplePrediction()
        print("fuck")'''

'''
plt.imshow(np.reshape(training_data_temp[0][0], [28,28]))
plt.show()'''

''''w,b = createNeuralNetwork(len(flattenedTrain[0]), 2, hiddenLay, 10)
w,b = trainNeuralNetwork(epochs, w, b, flattenedTrain, training_data_temp)

with open('w.pkl', 'wb') as outfile:
    pickle.dump(w, outfile, pickle.HIGHEST_PROTOCOL)

with open('b.pkl', 'wb') as outfile:
    pickle.dump(b, outfile, pickle.HIGHEST_PROTOCOL)'''
    
with open('w.pkl', 'rb') as infile:
    w = pickle.load(infile)
    
with open('b.pkl', 'rb') as infile:
    b = pickle.load(infile)

fullyConvSemSeg()
print("great success")
exit()

flattenedTrain = convolveImageSet(training_data_temp, 1, convs, convstep, pool)

print(len(flattenedTrain))

flattenedValid = convolveImageSet(test_data_temp, 1, convs, convstep, pool)

numCorrect = 0
avgAcc = 0
for i in range(len(flattenedValid)):
    tempX = flattenedValid[i]
    tempX = np.reshape(tempX, (1,len(tempX)))
    z,a = feedForward(tempX, w, b)

    x,y,z = calculateAccuracy(a[len(a)-1][0], test_data_temp[i][1].T[0])
    #print("answer ", x, " guess ", y, " certainty ", z)
    if x == y:
        numCorrect += 1
    avgAcc += z

print(numCorrect, numCorrect/trainSize, avgAcc/trainSize)
timeStop = datetime.datetime.now()
print(timeStop - timeStart)

exit()

flattenedTest = convolveImageSet(real_test_data, 1, convs, convstep, pool)

numCorrect = 0
avgAcc = 0
for i in range(len(flattenedTest)):
    tempX = flattenedTest[i]
    tempX = np.reshape(tempX, (1,len(tempX)))
    z,a = feedForward(tempX, w, b)

    x,y,z = calculateAccuracy(a[len(a)-1][0], [real_test_data[i][1]])
    #print("answer ", x, " guess ", y, " certainty ", z)
    if x == y:
        numCorrect += 1
    avgAcc += z

print(numCorrect, numCorrect/testSize, avgAcc/testSize)
timeStop = datetime.datetime.now()
print(timeStop - timeStart)