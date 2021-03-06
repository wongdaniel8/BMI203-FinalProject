from Bio import SeqIO
import random
import numpy as np
from sklearn import preprocessing, model_selection

testSet = []
full = []

def convertLabel(lab, arraySize):
    """
    converts a label (0 or 1) to a vector of length arraySize, with a 1 being in index {label}
    """
    lab = int(lab)
    ar = np.zeros(arraySize)
    ar[lab] = 1
    return ar

#preprocessing 
def getData(split=True):
    global testSet
    global full
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
        
        #maybe neural net will just pick up on if seq consists of repeats of length 17! 
        #get a random 17bp seq in the negative and repeat it
        rand = random.randint(0, len(seq) - 18)
        seq = seq[rand: rand + 17] 
        encoded = encode(seq, True) 
        
        # encoded = encode(seq, False)
       
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
        y.append(convertLabel(1, 2))

    randomNegs = random.sample(encodedNeg, 137) #training will be 274 total sequences (half positive half negative), negatives will be chosen randomly
    for el in randomNegs:
        X.append(el)
        y.append(convertLabel(0, 2))

    X = np.array(X, np.int32)
    y = np.array(y)

    #add fictitious dimension
    xFict = np.ones((len(X), 1001))
    xFict[:,:-1] = X
    X = np.copy(xFict)

    full = X

    # X = preprocessing.normalize(X, norm='l2')
    
    if split == False:
        return X, y
    else:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.10, random_state=42)
        return X_train, X_test, y_train, y_test


#===========================================================================================
class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 1000 + 1
        self.outputLayerSize = 2
        self.hiddenLayerSize = 200 + 1

        #define layers and activations
        self.x = [] #input layer
        self.h = [] #post-tanh values in hidden layer
        self.z = [] #post sigmoid function values in output layer
        self.u = [] #pre-tanh function values in hidden layer
        self.uPrime = [] #pre-sigmoid function values in ouput layer
        self.epsilon = .1
        #Weights (parameters)
        self.V = np.random.normal(0, .01, (self.inputLayerSize, self.hiddenLayerSize - 1))
        self.W = np.random.normal(0, .01, (self.hiddenLayerSize, self.outputLayerSize))

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
        batch will be true if we pass in a batch of sequences for X to compute y hat
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
        yHat = self.sigmoid(self.uPrime) 
        return yHat

    def costFunction(self, X, y):
        #Compute squared error cost for given X, y, use weights already stored in class.
        self.yHat = self.forward(X, True)
        J = 0.5 * np.sum(np.square(y - self.yHat))
        return J

    def costFunctionPrime(self, X, y, batch):
        """
        Compute derivative with respect to V and W for a given sample X and label y
        implement batch functionality later for batch gradient descent
        """
        if batch == False:
            self.yHat = self.forward(X, batch)
            delta3 = np.multiply(- (y - self.yHat), self.sigmoidPrime(self.uPrime))
            dJdW = np.dot(np.reshape(np.append(self.h, [1]), (1, self.hiddenLayerSize)).T, np.reshape(delta3, (1, self.outputLayerSize)))

            delta2 = np.dot(delta3, np.delete(self.W.T, np.s_[-1:], 1)) * (1 - (self.tanh(self.h))**2)####self.h does not have fict dimension
            dJdV = np.dot(np.reshape(X, (1, self.inputLayerSize)).T, np.reshape(delta2, (1, self.hiddenLayerSize - 1)))####
            return dJdV, dJdW #dV is mostly 0s

    def crossEntropyCost(self, X, y):
        self.yHat = self.forward(X, True)
        J = -1 * np.sum(y * np.log(self.yHat) + ((1 - y) * np.log(1 - self.yHat)))
        return J

    def crossEntropyCostPrime(self, X, y, batch):
        self.yHat = self.forward(X, batch)
        delta3 = -1 * np.multiply((y/self.yHat) - ((1-y)/(1-self.yHat)), self.sigmoidPrime(self.uPrime))
        dJdW = np.dot(np.reshape(np.append(self.h, [1]), (1, self.hiddenLayerSize)).T, np.reshape(delta3, (1, self.outputLayerSize)))
        delta2 = np.dot(delta3, np.delete(self.W.T, np.s_[-1:], 1)) * (1 - (self.tanh(self.h))**2)####self.h does not have fict dimension
        dJdV = np.dot(np.reshape(X, (1, self.inputLayerSize)).T, np.reshape(delta2, (1, self.hiddenLayerSize - 1)))####
        return dJdV, dJdW #dV is mostly 0s


    def trainNeuralNetwork(self, sequences, labels, validation, validationLabels, costType,verbose=True):
        errorRate = 1
        DONE = False
        iters = 0
        newThresh = .20
        trainingErrors = []
        itersList = []
        validationAccuracies = []
        costs = []
        cardinality = 274
        while iters < cardinality * 50: 
            #STOCHASTIC SINGLE sequence
            sample = sequences[iters % len(sequences)] 
            y = labels[iters % len(sequences)]
            if costType == "squared":
                dv, dw = self.costFunctionPrime(sample, y, False) 
            else:
                dv, dw = self.crossEntropyCostPrime(sample, y, False)
            self.V = self.V - float(self.epsilon) * dv #subtract every element in vector v by dv
            self.W = self.W - float(self.epsilon) * dw #subtract every element in vector w by dw
            iters += 1
           
            if errorRate < newThresh:
                self.epsilon = self.epsilon * .6
                newThresh -= .05
                DONE = True 

            if iters % 10 == 0:
                preds = self.predict("", validation)
                mistakes = 0
                for i in range(0, len(preds)):
                    if preds[i] != np.argmax(validationLabels[i]):
                        mistakes += 1
                errorRate = mistakes / float(len(preds))
                if verbose == True:
                    print(preds)
                    print("iteration: ", iters, "self.epsilon: ", self.epsilon, " val error rate: ", mistakes / float(len(preds)))
                validationAccuracies.append(1 - errorRate)
            
            if errorRate < .04:
                print("less than .04 validation error")
                avg = crossValidate(self, np.concatenate((sequences, validation)), np.concatenate((labels, validationLabels)), verbose=False)
                if avg < .047:
                    break

            if iters % 30 == 0:
                if costType == "squared":
                    cost = self.costFunction(sequences,labels)
                    costs.append(cost)
                else:
                    cost = self.crossEntropyCost(sequences, labels) 
                    costs.append(cost)
                if verbose == True:
                    print("COST", cost) 

        return self.V, self.W

    def predict(self, weights, sequences, getProbTrue=False):
        """
        returns predictions as integers (binary classification) if getProbTrue = False
        else will return predictions as a list of probabilities of being a 1 (positive result)
        """
        predictions = []
        for seq in sequences:
            pred = self.forward(seq, False)
            if getProbTrue:
                # predictions.append(pred[1]) 
                predictions.append(1 - pred[0]) #prob of not false = prob of being True (being positive), chosen this way because margin on left is slightly smaller
            else:
                predictions.append(np.argmax(pred)) 
        return predictions

def getNN(costType):
    training, validation, trainingLabels, validationLabels = getData(True)
    X = np.concatenate((training, validation),axis=0)
    y = np.concatenate((trainingLabels, validationLabels))
    if costType == "squared":
        NN = Neural_Network()
        NN.trainNeuralNetwork(training, trainingLabels, validation, validationLabels, "squared", verbose=False)
    else:
        NN = Neural_Network()
        NN.trainNeuralNetwork(training, trainingLabels, validation, validationLabels, "entropy", verbose=False)

    predictions = NN.predict("", validation)
    mistakes = 0
    for i in range(0, len(predictions)):
        if predictions[i] != np.argmax(validationLabels[i]):
            mistakes += 1
    print("final error rate on validation: ", mistakes / float(len(predictions)))
    print("final k fold")
    crossValidate(NN, X, y, False)
    return NN

def crossValidate(NN, X, y, verbose=False):
    """
    performs cross validation with an input neural network, prints results
    """
    folds = []
    kf = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        predictions = NN.predict("", X_test)
        mistakes = 0
        for i in range(0, len(predictions)):
            if predictions[i] != np.argmax(y_test[i]):
                mistakes += 1
        print("iter k fold error", mistakes / float(len(predictions)))
        folds.append(mistakes / float(len(predictions)))
    print("average k fold", np.average(folds))
    return np.average(folds)

def predictOnTestSet():
    """
    write output of test predictions to file "predictions.txt"
    """
    global testSet
    NN = getNN("squared")
    print("test on all ordered")
    print(NN.predict("", full))
    testPredictions = NN.predict("", testSet, getProbTrue=True)
    with open("predictions_new.txt", "w") as f:
        with open("rap1-lieb-test.txt", "r") as t:
            i = 0
            for line in t.readlines():
                f.write(line.rstrip('\n') + '\t' + str(testPredictions[i]) + '\n')
                i+=1

def autoencoder():
    """
    check correctness of neural network by feeding it an identity matrix of length 8, 
    see if it learns binary
    """
    NN = Neural_Network()
    NN.inputLayerSize = 8 
    NN.outputLayerSize = 8
    NN.hiddenLayerSize = 3
    NN.V = np.random.normal(0, .01, (NN.inputLayerSize, NN.hiddenLayerSize - 1))
    NN.W = np.random.normal(0, .01, (NN.hiddenLayerSize, NN.outputLayerSize))
    NN.epsilon = 5
    print(NN.V.shape, NN.W.shape)
    print(NN.inputLayerSize)
    X = np.identity(8) 
    y = X
    training, validation, trainingLabels, validationLabels = model_selection.train_test_split(X, y, test_size= 2, random_state=42)
    NN.trainNeuralNetwork(X, y, validation, validationLabels, "squared", verbose=True)
    predictions = NN.predict("", X)
    print("predictions (should be ordered 0->7):",predictions)
    return predictions


# autoencoder()
# NN = getNN("squared")
# crossValidate()
# predictOnTestSet()





#========================================================================
# Notes:
# 274 total size of set (training + validation test)
# predicitons_new2 has 0 error validation,  .04 k fold error 







