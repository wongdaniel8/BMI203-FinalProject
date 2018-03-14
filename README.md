# BMI203-FinalProject

[![Build
Status](https://travis-ci.org/wongdaniel8/BMI203-HW3.svg?branch=master)](https://travis-ci.org/wongdaniel8/BMI203-FinalProject)

travis: https://travis-ci.org/wongdaniel8/BMI203-FinalProject

Final project implementation for the the class BMI203 at UCSF
3 layer feed forward neural network to predict transcription factor binding sites of the protein Rap1
Performs optimally on holdout test set size of 10% of data

Positive (known to bind) sequences are in rap1-lieb-test.txt
Negative sequences are in yeast-upstream-1k-negative.fa

Data Representation in NN:
I encode the inputs to the neural network as follows. For the positive sequences, I convert the base pairs into numbers using the following encoding schema: {'A':1, 'C':2, 'G':3, 'T':4}. I then repeat the sequence until I have a length 1000 bp sequence. For the negative sequences, I take a random substring of the 1000 length sequence to get a substring of length 17. I then do the same encoding and repeating until I have a length 1000 sequence. See method getData(). The input layer has size 1000 (one for each feature, i.e. index in the sequence) and the hidden layer has size of 201 (arbitrarily chosen and seems to work well). The output layer has size of 2 nodes (one that embodies the positive class and the other the negative class) which measure the likelihood that the input data is this class. Output in this layer is in the range [0, 1] for probability. Argmax is used to determine the binary label as opposed to a continuous label.

For my data selection scheme, I first constructed a set of 274 sequences (all of the positive sequences), and 137 randomly chosen negatives sequences. Before selection of negative sequences, I filtered out the ones that contained a positive sequence. I chose to balance my positive and negative training examples to counter the massive selection bias of the negative sequences. From this set, I randomly took 90% of it and used for training the model, and 10% to use for testing (model does not see this data). Periodically while training, I’d predict on the test set. My stopping parameter was stopping after my test set accuracy dropped below 3.6% (i.e. one misclassified sequence per 28 sequences). My average k fold cross validation score dipped below 5% error. If this never happened during training, I also capped the max number of iterations over the training data to 50 epochs. I chose error < 3.6% for my test set because this was the best performance I could achieve with the data. 