import implicit
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from statistics import mean
from matrixFactorMethods import get_err, centerUV, calculateProjection


def implicitModel(movieLensDataTrainPath='train_clean.txt', movieLensDataTestPath='test_clean.txt'):
    ''' Implementation of the implicit model. Takes in train and testing data. '''

    # Load in training and testing data
    dfTrain = pd.read_csv(movieLensDataTrainPath, sep="\t", header=None)
    dfTrain.columns = ["User Id", "Movie Id", "Rating"]

    dfTest = pd.read_csv(movieLensDataTestPath, sep="\t", header=None)
    dfTest.columns = ["User Id", "Movie Id", "Rating"]

    test = dfTest.to_numpy()
    train = dfTrain.to_numpy()

    # Initialize a model based on the implicit model
    model = implicit.als.AlternatingLeastSquares(factors=25, iterations=400, regularization=0.01)
    # Declare M and N
    M = max(max(train[:, 0]), max(test[:, 0])).astype(int)
    N = max(max(train[:, 1]), max(test[:, 1])).astype(int) + 1

    # We need a matrix to store all values of Y since it
    # expects an actual M x N matrix with each i, j
    # entry containing Y_ij.
    newTrains = np.zeros((M, N))
    for y in train:
        i, j, yij = y
        i = i - 1
        j = j
        newTrains[i][j] = yij
    newTrains = np.array(newTrains)

    # Convert to a format accepted by
    # implicit.
    train = csr_matrix(newTrains)
    # Train the model on a sparse matrix of movie/user/confidence weights
    model.fit(train)
    # These are our corresponding U and V matrices
    return model.item_factors, model.user_factors

# Without this, the error goes up to around 6. Don't know why.
'''def get_err2(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    totalLength = len(Y)

    sumOfSqs = 0
    #meanYs = mean(Y[:, 2])
    for y in Y:
        #print(y)
        i = int(y[0])
        j = int(y[1])
        yij = y[2]
        i = i - 1
        j = j
        sumOfSqs = sumOfSqs + ((yij - np.dot(U[i], V[j])) ** 2)

    normSum = (np.linalg.norm(U, ord='fro') ** 2 + np.linalg.norm(V, ord='fro') ** 2)
    return ((reg * normSum) + sumOfSqs) / (2 * totalLength)'''


def Vtrain(Y, max_epochs=400):
    ''' This trains the matrix V on implicit to retrieve a matrix factorization. '''
    model = implicit.als.AlternatingLeastSquares(factors=25, iterations=max_epochs, regularization=0.01)
    # Train here
    newTrains = np.array(Y)
    train = csr_matrix(newTrains)
    #print(train.shape)
    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(train)

    print(model.item_factors.shape)
    # print(len(model.item_factors))
    print(model.user_factors.shape)
    # print(len(model.user_factors))
    return model.item_factors, model.user_factors

def SVDofV(oldV):
    ''' SVDofV() finds the SVD of V, using same method as before: through implicit. '''
    # Use to compute Ein and Eout
    A, B = Vtrain(oldV, max_epochs=300)
    return A, B


def tryThis():
    ''' Main engine, basically we didn't know how promising this was, so
        we just wanted to try it. The U, V is obtained, and then transposed
        to then collect the SVD of V. Then, the A is used for calculating
        projection. Then it is tested. '''
    U, V = implicitModel()
    U = np.float64(U)
    V = np.float64(V)
    U = U.T
    V = V.T
    # Center U and V.
    U, V = centerUV(U, V)

    # SVD of V!
    A, B = SVDofV(V)
    A = A.T
    # Use the first 2 cols for 2 dimensional projection.
    projU, projV = calculateProjection(A, U, V)
    dfTest = pd.read_csv('../data/test_clean.txt', sep="\t", header=None)
    dfTest.columns = ["User Id", "Movie Id", "Rating"]
    # Calculate error.
    Y_test = dfTest.to_numpy()
    print(get_err(U.T, V.T, Y_test))
    print(get_err(projU.T, projV.T, Y_test))
    return projU, projV

tryThis()