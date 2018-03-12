from Bio import SeqIO
import random
import numpy as np
import random
from scipy import io
from sklearn import preprocessing, model_selection
import sys
from matplotlib import pyplot as plt

testSet = []

def convertLabel(lab):
    """
    converts a label (0 or 1) to a vector of length 2, with a 1 being in index {label}
    """
    ar = np.zeros(2)
    ar[lab] = 1
    return ar

#preprocessing 
def getData(split=True):
    global testSet
    #get raw data into list of sequences
    positives = []
    with open("rap1-lieb-positives.txt") as f:
        for line in f.readlines():
            line = line.rstrip("\n")
            positives.append(line)

    negatives = []
    for seq_record in SeqIO.parse("yeast-upstream-1k-negative.fa", "fasta"):
        seq1 = seq_record.seq
        if len(seq1) != 1000:
            continue
        negatives.append(str(seq1))

    tests = []
    with open("rap1-lieb-test.txt", 'r') as f:
        for line in f.readlines():
            tests.append(line.rstrip('\n'))


    #purge negatives that have a positive seq in it
    cleanedNegatives = []
    for negSeq in negatives:
        contained = False
        for posSeq in positives:
            if posSeq in negSeq:
                contained = True
                break
        if not contained:
            cleanedNegatives.append(negSeq)
    negatives = cleanedNegatives #3164 sequences to 3099

    #write negatives to text file
    with open("cleaned_negatives.txt", "w") as f:
        for neg in negatives:
            f.write(str(neg))
            f.write("\n")

    def encode(seq, isShort):
        mapp = {'A':1, 'C':2, 'G':3, 'T':4}
        l = []
        for char in seq:
            l.append(mapp[char])
        
        if isShort:
            l *= 58
            l += l[0:14]
        return l

    #encode sequences as vectors 
    encodedPos = []
    for seq in positives:
        encoded = encode(seq, True)
        encodedPos.append(encoded)

    encodedNeg = []
    for seq in negatives:
        encoded = encode(seq, False)
        encodedNeg.append(encoded)

    #get test set into same format 
    encodedTests = []
    for seq in tests:
        encoded = encode(seq, True)
        encodedTests.append(encoded)
    encodedTests = np.array(encodedTests, np.int32)
    xFict = np.ones((len(encodedTests), 1001))
    xFict[:,:-1] = encodedTests
    testSet = np.copy(xFict)

    #set up for training and validation    
    X = []
    y = []
    for el in encodedPos:
        X.append(el)
        y.append(convertLabel(1))

    randomNegs = random.sample(encodedNeg, 137) #137#training will be 274 total sequences (half positive half negative), negatives will be chosen randomly
    for el in randomNegs:
        X.append(el)
        y.append(convertLabel(0))

    X = np.array(X, np.int32)
    y = np.array(y)

    #add fictitious dimension
    xFict = np.ones((len(X), 1001))
    xFict[:,:-1] = X
    X = np.copy(xFict)

    # X = preprocessing.normalize(X, norm='l2')

    
    if split == False:
        return X, y
    else:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.1, random_state=42)
        return X_train, X_test, y_train, y_test


#===========================================================================================
class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 1000 + 1
        self.outputLayerSize = 2
        self.hiddenLayerSize =  200 + 1

        #define layers and activations
        self.x = [] #input layer
        self.h = [] #post-tanh values in hidden layer
        self.z = [] #post sigmoid function values in output layer
        self.u = [] #pre-tanh function values in hidden layer
        self.uPrime = [] #pre-sigmoid function values in ouput layer

        #Weights (parameters)
        self.V = np.random.normal(0, .01, (1001, self.hiddenLayerSize - 1))
        self.W = np.random.normal(0, .01, (self.hiddenLayerSize, 2))
        # self.V = np.load("V3.npy")
        # self.W = np.load("W3.npy")

    def sigmoid(self, z):
    #Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
    #Derivative of sigmoid function
        return np.exp(-z) / ((1 + np.exp(-z))**2)

    # def sigmoid(self, z):
    #     #RELU
    #     return np.log(1 + np.exp(z))

    # def sigmoidPrime(self, z):
    #     #RELU
    #     return 1 / (1 + np.exp(-z))
    
    def tanh(self, z):
        return np.tanh(z)

    def tanhPrime(self, z):
        return 1 - (self.tanh(z))**2
        
    def forward(self, X, batch):
        """
        batch will be true if we pass in a batch of images for X to compute y hat
        """
        #Propagate inputs though network
        self.u = np.dot(X, self.V)
        self.h = self.tanh(self.u)
        # self.uPrime = np.dot(hDim, self.W)
        hcopy = np.copy(self.h)
        if batch == True:
            hFict = np.ones((len(self.h), self.hiddenLayerSize))
            hFict[:, :-1] = hcopy
            self.uPrime = np.dot(hFict, self.W)
        else:
            self.uPrime = np.dot(np.append(hcopy, [1]), self.W)
        # self.uPrime = np.dot(self.h, self.W)
        yHat = self.sigmoid(self.uPrime) 
        # print(yHat.shape, yHat)
        return yHat

    def costFunction(self, X, y):
        #Compute squared error cost for given X, y, use weights already stored in class.
        self.yHat = self.forward(X, True)
        # preds = np.amax(self.yHat, axis=1) #preds will be the 1d array containing the max elements of each forward return
        J = 0.5 * np.sum(np.square(y - self.yHat))
        return J

    def costFunctionPrime(self, X, y, batch):
        """
        Compute derivative with respect to V and W for a given sample X and label y
        implement batch functionality later for batch gradient descent
        """
        # dJdW = 201x2 # dJdV = 1001x201  # delta2 = 1x201 but want 1x200???
         # dJdV = want 1001x200 
        if batch == False:
            self.yHat = self.forward(X, batch)
            delta3 = np.multiply(- (y - self.yHat), self.sigmoidPrime(self.uPrime))
            dJdW = np.dot(np.reshape(np.append(self.h, [1]), (1, self.hiddenLayerSize)).T, np.reshape(delta3, (1, 2)))

            delta2 = np.dot(delta3, np.delete(self.W.T, np.s_[-1:], 1)) * (1 - (self.tanh(self.h))**2)####self.h does not have fict dimension
            dJdV = np.dot(np.reshape(X, (1, 1001)).T, np.reshape(delta2, (1, self.hiddenLayerSize - 1)))####
            return dJdV, dJdW #dV is mostly 0s

    def crossEntropyCost(self, X, y):
        self.yHat = self.forward(X, True)
        J = -1 * np.sum(y * np.log(self.yHat) + ((1 - y) * np.log(1 - self.yHat)))
        return J

    def crossEntropyCostPrime(self, X, y, batch):
        self.yHat = self.forward(X, batch)
        delta3 = -1 * np.multiply((y/self.yHat) - ((1-y)/(1-self.yHat)), self.sigmoidPrime(self.uPrime))
        dJdW = np.dot(np.reshape(np.append(self.h, [1]), (1, self.hiddenLayerSize)).T, np.reshape(delta3, (1, 2)))
        delta2 = np.dot(delta3, np.delete(self.W.T, np.s_[-1:], 1)) * (1 - (self.tanh(self.h))**2)####self.h does not have fict dimension
        dJdV = np.dot(np.reshape(X, (1, 1001)).T, np.reshape(delta2, (1, self.hiddenLayerSize - 1)))####
        return dJdV, dJdW #dV is mostly 0s


    def trainNeuralNetwork(self, images, labels, validation, validationLabels, costType):
        # print("V: ", self.V)
        # print("W: ", self.W)
        # print("lengths: ", len(images), len(labels), len(validation))
        errorRate = 1
        decayStart = -1
        DONE = False
        iters = 0
        epsilon =  0.1
        trainingErrors = []
        itersList = []
        validationAccuracies = []
        costs = []
        cardinality = 274
        while iters < cardinality * 1000: 
            # print('xiters', iters)
            #STOCHASTIC SINGLE IMAGE
            sample = images[iters % len(images)] 
            y = labels[iters % len(images)] 
            if costType == "squared":
                dv, dw = self.costFunctionPrime(sample, y, False) 
            else:
                dv, dw = self.crossEntropyCostPrime(sample, y, False)
            self.V = self.V - float(epsilon) * dv #subtract every element in vector v by dv
            self.W = self.W - float(epsilon) * dw #subtract every element in vector w by dw
            iters += 1
            decayStart += 1
            
            if DONE == False and errorRate < .20:
                epsilon = epsilon * .6
                DONE = True
                decayStart = 1


            # if DONE == True and (decayStart % (5 * cardinality) == 0):
            #     epsilon = epsilon * .6
            
            # if costType == "entropy" and errorRate < .11:
            #     epsilon = .01
           
            if iters % 10 == 0:
            # if iters % cardinality == 0:
                preds = self.predict("", validation)
                print(preds)
                mistakes = 0
                for i in range(0, len(preds)):

                    if preds[i] != np.argmax(validationLabels[i]):
                        mistakes += 1
                errorRate = mistakes / float(len(preds))
                print("iteration: ", iters, "epsilon: ", epsilon, " val error rate: ", mistakes / float(len(preds)))
                validationAccuracies.append(1 - errorRate)
            
            if errorRate < .0003:
                print("less than .0003 error")
                break

            if iters % 10 == 0:
            # if iters % cardinality  == 0:
                preds = self.predict("", images)
                mistakes = 0
                for i in range(0, len(preds)):
                    if preds[i] != np.argmax(labels[i]):
                        mistakes += 1
                errorRate = mistakes / float(len(preds))
                print("iteration: ", iters, "epsilon: ", epsilon, " training error rate: ", mistakes / float(len(preds)))
                trainingErrors.append(errorRate)
                itersList.append(iters)

            # if iters % cardinality == 0:
            if iters % 10 == 0:
                if costType == "squared":
                    cost = self.costFunction(images, labels)
                    costs.append(cost)
                else:
                    cost = self.crossEntropyCost(images, labels) #PASSING IN ALL IMAGES TO COMPUTE COST
                    costs.append(cost)
                print("COST", cost) 

            if iters % cardinality == 0:
                images, validation, labels, validationLabels = getData(True)

        # plt.xlabel("iteration")
        # plt.ylabel("cost")
        # plt.plot(itersList, costs, 'ro')
        # plt.show()

        # plt.xlabel("iteration")
        # plt.ylabel("training error")
        # plt.plot(itersList, trainingErrors, 'ro')
        # plt.show()

        # plt.xlabel("iteration")
        # plt.ylabel("validation accuracy")
        # plt.plot(itersList, validationAccuracies, 'ro')
        # plt.show()

        
        return self.V, self.W

    def predict(self, weights, images, getProbTrue=False):
        """
        returns predictions as integers (binary classification) if getProbTrue = False
        else will return predictions as a list of probabilities of being a 1 (positive result)
        """
        predictions = []
        # counts = 0
        for image in images:
            pred = self.forward(image, False)
            # print(pred,1 - float(pred[1]) , 1 - float(pred[1]) > pred[0], 1 - float(pred[1]) - pred[0])
            # if (1 - float(pred[1]) > pred[0]):
            #     counts += 1
            if getProbTrue:
                # predictions.append(pred[1]) 
                predictions.append(1 - pred[0]) #prob of not false = prob of being True (being positive), chosen this way because margin on left is slightly smaller
            else:
                predictions.append(np.argmax(pred)) #arg min? 
        # print("COUNTS", counts/float(len(predictions)))
        return predictions
        



def validate(costType):

    training, validation, trainingLabels, validationLabels = getData(True)
    if costType == "squared":
        NN = Neural_Network()
        NN.trainNeuralNetwork(training, trainingLabels, validation, validationLabels, "squared")
    else:
        NN = Neural_Network()
        NN.trainNeuralNetwork(training, trainingLabels, validation, validationLabels, "entropy")

    predictions = NN.predict("", validation)
    mistakes = 0
    for i in range(0, len(predictions)):
        if predictions[i] != np.argmax(validationLabels[i]):
            mistakes += 1
    print("error rate: ", mistakes / float(len(predictions)))
    return NN


def crossValidate():

    X, y = getData(False)
    kf = model_selection.KFold(n_splits=10, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        NN = Neural_Network()
        NN.trainNeuralNetwork(X_train, y_train, X_test, y_test, "squared")
        predictions = NN.predict("", X_test)
        mistakes = 0
        for i in range(0, len(predictions)):
            if predictions[i] != np.argmax(y_test[i]):
                mistakes += 1
        print("k fold error rate: ", mistakes / float(len(predictions)))



def predictOnTestSet():
    """
    write output of test predictions to file "predictions.txt"
    """
    global testSet
    training, validation, trainingLabels, validationLabels = getData(True)
    NN = Neural_Network()
    NN.trainNeuralNetwork(training, trainingLabels, validation, validationLabels, "squared")
    testPredictions = NN.predict("", testSet, getProbTrue=True)
    with open("predictions.txt", "w") as f:
        with open("rap1-lieb-test.txt", "r") as t:
            i = 0
            for line in t.readlines():
                f.write(line.rstrip('\n') + '\t' + str(testPredictions[i]) + '\n')
                i+=1

# NN = validate("squared")
# crossValidate()
predictOnTestSet()












