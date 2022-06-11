from mnist import MNIST
import numpy as np
from random import randrange
import random
import math
import pickle
import time
from datetime import datetime
#from PIL import Image
from collections import Iterable
from grid import CreateRoster

mndata = MNIST("C:\\Users\\Jim\\Desktop\\NeuronalesNetz")
trainimages, trainlabels = mndata.load_training()
testimages, testlabels = mndata.load_testing()
"""
index = randrange.randrange(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))
print(labels[index])
print(images[index])
"""


def sigmoid(x, ableitung):
    # return max(0.0, x)
    if ableitung:
        return sigmoid(x, False) * (1 - sigmoid(x, False))
    else:
        return 1 / (1 + math.e ** -x)


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def loadClassifier(file):
    pickleIn = open(file, "rb")
    returnClass = pickle.load(pickleIn)
    pickleIn.close()
    return returnClass


class NeuralNet:
    def __init__(self, layerslist, biasrange, weightrange):
        self.layerlist = layerslist
        self.layernumber = len(layerslist)
        self.layers = [np.array(layer * [0]) for layer in layerslist]
        self.layerZ = [np.array(layer * [0]) for layer in layerslist]
        self.layerZ = self.layerZ[1:]
        self.layerbiasis = [
            np.random.uniform(-biasrange, biasrange, size=(layer))
            for layer in layerslist
        ]
        self.layerbiasis = self.layerbiasis[1:]
        self.layerweights = [
            np.random.uniform(
                -weightrange, weightrange, size=(layerslist[i + 1], layerslist[i])
            )
            for i in range(self.layernumber - 1)
        ]
        self.layerbiasischange = [0 * biasis for biasis in self.layerbiasis]
        self.layerweightschange = [0 * weights for weights in self.layerweights]

    def saveClassifier(self, file):
        pickleOut = open(file, "wb")
        pickle.dump(self, pickleOut)
        pickleOut.close()

    def getOutput(self, input):
        self.layers[0] = 1 / 255 * np.array(input)
        for i in range(0, self.layernumber - 1):
            self.layerZ[i] = self.layerweights[i] @ self.layers[i] + self.layerbiasis[i]
            self.layers[i + 1] = sigmoid(self.layerZ[i], False)

    def backpropagation(self, wantedOutput):
        self.layerbiasischange[-1] = (
            1 * sigmoid(self.layerZ[-1], True) * 2 * (self.layers[-1] - wantedOutput)
        )
        for i in range(-2, -self.layernumber, -1):
            self.layerbiasischange[i] = sigmoid(self.layerZ[i], True) * (
                np.transpose(self.layerweights[i + 1]) @ self.layerbiasischange[i + 1]
            )

        for i in range(-1, -self.layernumber + 1, -1):
            self.layerweightschange[i] = np.array(
                self.layerlist[i] * [self.layers[i - 1]]
            ) * self.layerbiasischange[i].reshape(self.layerlist[i], 1)

        for i in range(0, self.layernumber - 1):
            self.layerbiasischangebatch[i] -= self.layerbiasischange[i]
            self.layerweightschangebatch[i] -= self.layerweightschange[i]
            # self.layerbiasis[i] -= self.layerbiasischange[i]
            # self.layerweights[i] -= self.layerweightschange[i]

    def trainNetwork(self, images, labels, batch):
        for i in range(0, int(60000 / batch)):
            self.layerbiasischangebatch = [0 * biasis for biasis in self.layerbiasis]
            self.layerweightschangebatch = [
                0 * weights for weights in self.layerweights
            ]
            for ii in range(0, batch):
                index = i * batch + ii
                self.wantedOutput = np.array(
                    labels[index] * [0] + [1] + (9 - labels[index]) * [0]
                )
                self.getOutput(images[index])
                self.backpropagation(self.wantedOutput)

            for ii in range(0, self.layernumber - 1):
                self.layerbiasis[ii] += self.layerbiasischangebatch[ii] / batch
                self.layerweights[ii] += self.layerweightschangebatch[ii] / batch

    def testNetwork(self, images, labels):
        count = 0
        for index in range(0, 10000):
            self.getOutput(images[index])
            if (
                np.where(self.layers[-1] == np.amax(self.layers[-1]))[0][0]
                == labels[index]
            ):
                count += 1
        return count / 10000

    def testimage(self, image):
        print(mndata.display(image))
        self.getOutput(image)
        return np.where(self.layers[-1] == np.amax(self.layers[-1]))[0][0]

    def calculateLoss(self, images, labels):
        loss = 0
        for index in range(0, 60000):
            self.getOutput(images[index])
            self.wantedOutput = np.array(
                labels[index] * [0] + [1] + (9 - labels[index]) * [0]
            )
            loss += sum((self.layers[-1] - self.wantedOutput) ** 2)
        return loss / 60000


def trainIntereNetwork(ainumber):
    # start = time.time()
    for i in range(0, 3):
        liste = list(zip(trainimages, trainlabels))
        random.shuffle(liste)
        images, labels = zip(*liste)
        NeuralNet1.trainNetwork(images, labels, 100)
        print(i)
    with open("timestamps.txt", "a+") as file:
        file.write(
            f"AInumber: {ainumber}, test result: {NeuralNet1.testNetwork(testimages, testlabels)}\n"
        )


def test():
    index = randrange(0, len(trainimages))
    wantedOutput = np.array(
        trainlabels[index] * [0] + [1] + (9 - trainlabels[index]) * [0]
    )
    for i in range(0, 15):
        NeuralNet1.getOutput(trainimages[index])
        NeuralNet1.backpropagation(wantedOutput)
        print(NeuralNet1.layers[-1])
        print(sum((NeuralNet1.layers[-1] - wantedOutput) ** 2))
    print(wantedOutput)


# NeuralNet1 = loadClassifier("savedClassifier65")
# trainIntereNetwork(65)
# print(NeuralNet1.layerbiasis)
# print(NeuralNet1.layerweights)
# index = randrange(0, len(testimages))
# print(NeuralNet1.testimage(testimages[index]))
# NeuralNet1.printNeuralNet()
# test()
# NeuralNet1.saveClassifier(f"savedClassifier65")
"""
for i in range(40, 80):
    hiddenlayer2 = randrange(15, 30)
    hiddenlayer1 = randrange(1, 4) * hiddenlayer2
    biasrange = randrange(1, 14)
    weightrange = random.uniform(0.5, 1.5)
    NeuralNet1 = NeuralNet(
        [784, hiddenlayer1, hiddenlayer2, 10], biasrange, weightrange
    )
    trainIntereNetwork(i)
    with open("timestamps.txt", "a+") as file:
        file.write(
            f"hiddenlayer1: {hiddenlayer1}, hiddenlayer2: {hiddenlayer2}, biasrange: {biasrange}, weightrange: {weightrange}, loss: {NeuralNet1.calculateLoss(trainimages, trainlabels)}, filenumber: {i}\n"
        )
    NeuralNet1.saveClassifier(f"savedClassifier{i}")
"""
NeuralNet1 = loadClassifier("savedClassifier65")
# bild = Image.open("testbild1.png")
# bild1 = np.array(bild)
# bild1 = [[ii[0] for ii in i] for i in bild1]
# bild1 = list(flatten(bild1))
"""
bild1 = CreateRoster().main()
print(bild1)
bild2 = []
for x in range(0, 28):
    for y in range(0, 28):
        bild2 += [bild1[28 * y + x]]
"""
bild2 = testimages[randrange(0, 10000)]
print(NeuralNet1.testimage(bild2))
NeuralNet1.saveClassifier(f"savedClassifier65")