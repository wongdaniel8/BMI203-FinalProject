import numpy as np
import neuralNet

def testNN():
    """
    check if NN can learn binary
    """
    assert np.array_equal([0,1,2,3,4,5,6,7], neuralNet.autoencoder())
