from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt



# Method 1: Off the shelf and recommended method for the class.
def surpriseSVD(movieLensDataPath = '../data/data.txt'):
    ''' Basic use of the surprise SVD algorithm. '''
    ''' Params: movieLensDataPath is the path to the movielens data we're looking at. '''
    ''' Note: replace with cleaned data. '''
    ''' We want to return U and V where for a Y of a matrix of movie ratings, Y ~/= U^TV.'''
    # Load the data as a pandas data frame, as reading from text didn't quite work at first.
    df = pd.read_csv(movieLensDataPath, sep="\t", header=None)
    df.columns = ["User Id", "Movie Id", "Rating"]

    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))

    # The columns are User Id, Movie Id, and Rating.
    data = Dataset.load_from_df(df[["User Id", "Movie Id", "Rating"]], reader)

    # Use the famous SVD algorithm. To fit, we have to convert it to a trainset.
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    # Return U (qi) and V (pu)
    return algo.qi, algo.pu

# Method 2 (next functions): Seem familiar? I just ran this method for an epoch amount
# with default parameters. I will probably play around with this a bit
# and take a closer look at the slides.

def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - (Vj * (Yij - np.dot(Ui, Vj))))


def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - (Ui * (Yij - np.dot(Ui, Vj))))


def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    totalLength = len(Y)

    sumOfSqs = 0

    for y in Y:
        i, j, yij = y
        i = i - 1
        j = j - 1
        sumOfSqs = sumOfSqs + ((yij - np.dot(U[i], V[j])) ** 2)

    normSum = (np.linalg.norm(U, ord='fro') ** 2 + np.linalg.norm(V, ord='fro') ** 2)
    return ((reg * normSum) + sumOfSqs) / (2 * totalLength)


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    np.seterr(all='raise')

    U = np.random.uniform(low=-0.5, high=0.5, size=(M, K))
    V = np.random.uniform(low=-0.5, high=0.5, size=(N, K))
    step = get_err(U, V, Y, reg=reg)
    err = step
    print("K: " + str(K))
    for epoch in range(max_epochs):
        # Random permutation of the array
        # epoch_U = np.random.permutation(Y)
        yinds = np.random.permutation(len(Y))
        # For each point, perform gradient weight update.
        for ind in yinds:
            i, j, yij = Y[ind]
            i = i - 1
            j = j - 1
            newU = grad_U(U[i, :], yij, V[j, :], reg, eta)
            newV = grad_V(V[j, :], yij, U[i, :], reg, eta)

            # Sanity checks
            # eta (step size) * gradient result
            # Another sanity check where we
            # break if the results were wrong.
            U[i, :] = U[i, :] - newU
            V[j, :] = V[j, :] - newV
        newErr = get_err(U, V, Y, reg=reg)

        if epoch >= 1:
            fract = abs(err - newErr) / step
            err = newErr
            if fract < eps:
                print("Stopping at epoch: " + str(epoch))
                print("Error is: " + str(err))
                break
        else:
            err = newErr
            if epoch == 0:
                step = abs(err - step)

    return U, V, err

def naiveMinimization(movieLensDataTrainPath='../data/train.txt', movieLensDataTestPath='../data/test.txt'):
    dfTrain = pd.read_csv(movieLensDataTrainPath, sep="\t", header=None)
    dfTrain.columns = ["User Id", "Movie Id", "Rating"]

    dfTest = pd.read_csv(movieLensDataTestPath, sep="\t", header=None)
    dfTest.columns = ["User Id", "Movie Id", "Rating"]

    Y_test = dfTest.to_numpy()
    Y_train = dfTrain.to_numpy()
    M = max(max(Y_train[:, 0]), max(Y_test[:, 0])).astype(int)  # users
    N = max(max(Y_train[:, 1]), max(Y_test[:, 1])).astype(int)  # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    #Ks = [10, 20, 30, 50, 100]
    K = 10
    reg = 0.0
    eta = 0.03  # learning rate


    # Use to compute Ein and Eout
    U, V, err = train_model(M, N, K, eta, reg, Y_train)
    print(err)
    print(get_err(U, V, Y_test))
    return U, V

naiveMinimization()