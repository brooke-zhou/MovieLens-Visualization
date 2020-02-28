from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.decomposition import PCA, TruncatedSVD
#import implicit


def centerUV(U, V):
    '''
    Center V, U as each row should have 0 mean.
    '''
    # A TA said this was a good idea.
    for i in range(len(V)):
        V[i] = V[i] - mean(V[i])
    for i in range(len(U)):
        U[i] = U[i] - mean(U[i])
    return U, V

def calculateProjection(A, U, V):
    '''
    Calculate the two dimensional projections of
    U and V using A!
    '''
    # Use the first 2 cols for work
    Asub = A[:, :2]

    # Calculate the two dimensional projection!
    projU = np.dot(Asub.T, U)
    projV = np.dot(Asub.T, V)

    # Rescale dimensions to compress the image
    for i in range(len(projV)):
        projV[i] = projV[i] / max(projV[i])
    for i in range(len(projU)):
        projU[i] = projU[i] / max(projU[i])
    return projU, projV

# Method 1: Original SVD
def grad_U1(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.

    Note: grad_U1 does not include bias terms!
    """
    return eta * (reg * Ui - (Vj * (Yij - np.dot(Ui, Vj))))

def grad_V1(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.

    Note: grad_V1 does not include bias terms!
    """
    return eta * (reg * Vj - (Ui * (Yij - np.dot(Ui, Vj))))

# Culprit earlier
def get_err2(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.

    Note: get_err2 does not include bias terms!
    """
    totalLength = len(Y)
    sumOfSqs = 0
    # Do we need meanYs?
    #meanYs = mean(Y[:, 2])
    # Compute squared error
    for y in Y:
        i = int(y[0])
        j = int(y[1])
        yij = y[2]
        i = i - 1

        #j = j
        sumOfSqs = sumOfSqs + ((yij - np.dot(U[i], V[j])) ** 2)
    # Compute the regularized component.
    normSum = (np.linalg.norm(U, ord='fro') ** 2 + np.linalg.norm(V, ord='fro') ** 2)
    # Returns the mean regularized squared-error
    return ((reg * normSum) + sumOfSqs) / (2 * totalLength)

def SVDofV(oldV, K=20, max_epochs=300):
    ''' SVDofV() finds the SVD of V through the same method as the SVD '''
    ''' from homework 5. The main difference is that we already have '''
    ''' the matrix we want to factorize, while we have to build that '''
    ''' for Y from the dataset. And we only return A and B. '''
    # Find our dimensions M and N!
    M = len(oldV)
    N = len(oldV[0])
    reg = 0.0
    eta = 0.03  # learning rate
    Y = []
    for i in range(len(oldV)):
        for j in range(len(oldV[i])):
            Y.append([i, j, oldV[i][j]])
    Y = np.array(Y)
    # Use to compute Ein and Eout
    A, B, err = Vtrain_model(M, N, K, eta, reg, Y, max_epochs=max_epochs)
    return A, B

def Vtrain_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
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

    Note: this version has no biases terms and was originally used
    to get an approximate decomposition of V.
    """
    np.seterr(all='raise')
    # Initialize U and V to be matrices with random uniform
    # numbers of small magnitude.
    U = np.random.uniform(low=-0.5, high=0.5, size=(M, K))
    V = np.random.uniform(low=-0.5, high=0.5, size=(N, K))

    step = get_err2(U, V, Y, reg=reg)
    err = step

    for epoch in range(max_epochs):
        # Random permutation of the array
        yinds = np.random.permutation(len(Y))
        #meanYs = mean(Y[:, 2])
        # For each point, perform gradient weight update.
        for ind in yinds:
            # Unpack Y_ij, i, and j.
            y = Y[ind]
            i = int(y[0])
            j = int(y[1])
            yij = y[2]
            i = i - 1
            #j = j

            # Compute the gradients
            newU = grad_U1(U[i, :], yij, V[j, :], reg, eta)
            newV = grad_V1(V[j, :], yij, U[i, :], reg, eta)

            # Update U and V simultaneously
            U[i, :] = U[i, :] - newU
            V[j, :] = V[j, :] - newV

        # Compute the train error.
        newErr = get_err2(U, V, Y, reg=reg)

        # Stop if the decrease of error is too
        # low or is at most 0.
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

    # The definition changed from Problem Set 5, so we're pushing
    # the miniproject definition of U and V.
    return U.T, V.T, err


def originalSVD(movieLensDataTrainPath='../data/train_clean.txt', movieLensDataTestPath='../data/test_clean.txt'):
    ''' originalSVD() is the main engine of original SVD! It grabs the data, '''
    ''' calculates U and V from Y ~= U^TV, and then calculates SVD of V. '''
    ''' This is used to calculate the 2 dimension projection of U and V, which '''
    ''' are returned. '''

    # Read in train and test data!
    dfTrain = pd.read_csv(movieLensDataTrainPath, sep="\t", header=None)
    dfTrain.columns = ["User Id", "Movie Id", "Rating"]

    dfTest = pd.read_csv(movieLensDataTestPath, sep="\t", header=None)
    dfTest.columns = ["User Id", "Movie Id", "Rating"]

    Y_test = dfTest.to_numpy()
    Y_train = dfTrain.to_numpy()

    M = max(max(Y_train[:, 0]), max(Y_test[:, 0])).astype(int)  # users
    N = max(max(Y_train[:, 1]), max(Y_test[:, 1])).astype(int) + 1  # movies

    K = 10
    print("K: " + str(K))
    reg = 0.0
    eta = 0.03  # learning rate

    # Use to compute Ein and Eout
    U, V, err = Vtrain_model(M, N, K, eta, reg, Y_train, max_epochs=0)
    print("In sample")
    print(err)

    U, V = centerUV(U, V)

    # SVD of V!
    A, B = SVDofV(V, K=K, max_epochs=0)

    projU, projV = calculateProjection(A, U, V)
    print("Out of sample")
    print(get_err2(U.T, V.T, Y_test))
    print(get_err2(projU.T, projV.T, Y_test))
    return projU, projV

# Method 3 (next functions): Seem familiar?
# This is specifically the method that accounts for
# biases.
# originalSVDWithBellsWhistles() is the main one to run!
def grad_U(Ui, Yij, Vj, meanYs, a_i, b_j, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), mean of all the Ys, bias value a_i, bias value
    b_j, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta * (reg * Ui - (Vj * (Yij - meanYs - np.dot(Ui, Vj) - a_i - b_j)))


def grad_V(Vj, Yij, Ui, meanYs, a_i, b_j, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), mean of all the Ys, bias value a_i, bias value
    b_j, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta * (reg * Vj - (Ui * (Yij - meanYs - np.dot(Ui, Vj) - a_i - b_j)))

def grad_a(Ui, Yij, Vj, meanYs, a_i, b_j, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), mean of all the Ys, bias value a_i, bias value
    b_j, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to a_i multiplied by eta.
    """
    return eta * (reg * a_i - (Yij - meanYs - np.dot(Ui, Vj) - a_i - b_j))

def grad_b(Vj, Yij, Ui, meanYs, a_i, b_j, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), mean of all the Ys, bias value a_i, bias value
    b_j, reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to b_j multiplied by eta.
    """
    return eta * (reg * b_j - (Yij - meanYs - np.dot(Ui, Vj) - a_i - b_j))

def get_err(U, V, Y, a, b, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V, and bias vectors a and b.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    totalLength = len(Y)
    sumOfSqs = 0
    # Take the mean of Ys
    meanYs = mean(Y[:, 2])
    # Compute squared error
    for y in Y:
        i = int(y[0])
        j = int(y[1])
        yij = y[2]
        i = i - 1
        # Remove - 1
        #j = j - 1
        sumOfSqs = sumOfSqs + ((yij - meanYs - np.dot(U[i], V[j]) - a[i] - b[j]) ** 2)
    # Compute the regularized component.
    normSum = (np.linalg.norm(U, ord='fro') ** 2 + np.linalg.norm(V, ord='fro') ** 2 + \
               np.linalg.norm(a, ord='fro') ** 2 + np.linalg.norm(b, ord='fro') ** 2)
    # Returns the mean regularized squared-error
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
    # Initialize U, V, a, and b to be matrices with random uniform
    # numbers of small magnitude.
    U = np.random.uniform(low=-0.5, high=0.5, size=(M, K))
    V = np.random.uniform(low=-0.5, high=0.5, size=(N, K))
    a = np.random.uniform(low=-0.2, high=0.2, size=(M, 1))
    b = np.random.uniform(low=-0.2, high=0.2, size=(N, 1))
    step = get_err(U, V, Y, a, b, reg=reg)
    err = step
    # We perform at most max_epochs updates.
    for epoch in range(max_epochs):
        # Random permutation of the array
        yinds = np.random.permutation(len(Y))
        meanYs = mean(Y[:, 2])
        # For each point in Y, perform gradient weight update.
        for ind in yinds:
            y = Y[ind]
            i = int(y[0])
            j = int(y[1])
            yij = y[2]
            i = i - 1
            # Remove - 1
            #j = j - 1

            # Calculate all the gradients.
            newU = grad_U(U[i, :], yij, V[j, :], meanYs, a[i], b[j], reg, eta)
            newV = grad_V(V[j, :], yij, U[i, :], meanYs, a[i], b[j], reg, eta)
            newa = grad_a(U[i, :], yij, V[j, :], meanYs, a[i], b[j], reg, eta)
            newb = grad_b(V[j, :], yij, U[i, :], meanYs, a[i], b[j], reg, eta)

            # Update U, V, a, and b independently as gradients
            # were calculated in the previous step.
            U[i, :] = U[i, :] - newU
            V[j, :] = V[j, :] - newV
            a[i] = a[i] - newa
            b[j] = b[j] - newb

        # Calculate training error.
        newErr = get_err(U, V, Y, a, b, reg=reg)

        # Check for the stopping condition.
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

    return U, V, err, a, b


def naiveMinimization(movieLensDataTrainPath='train_clean.txt', movieLensDataTestPath='test_clean.txt',K=10):
    ''' Calculate SVD using biases. The logic should be same as originalSVD(). '''

    # Load the train and test data sets.
    dfTrain = pd.read_csv(movieLensDataTrainPath, sep="\t", header=None)
    dfTrain.columns = ["User Id", "Movie Id", "Rating"]
    dfTest = pd.read_csv(movieLensDataTestPath, sep="\t", header=None)
    dfTest.columns = ["User Id", "Movie Id", "Rating"]
    Y_test = dfTest.to_numpy()
    Y_train = dfTrain.to_numpy()

    # Find M and N, and define other parameters.
    M = max(max(Y_train[:, 0]), max(Y_test[:, 0])).astype(int)  # users
    # add 1
    N = max(max(Y_train[:, 1]), max(Y_test[:, 1])).astype(int) + 1 # movies
    # Ks = [10, 20, 30, 50, 100]
    #K = 10
    reg = 0.0
    eta = 0.03  # learning rate

    # Train the model and return U and V.
    U, V, err, a, b = train_model(M, N, K, eta, reg, Y_train, max_epochs=300)
    # Calculate errors, training and testing.
    print(err)
    print(get_err(U, V, Y_test, a, b))
    return U.T, V.T, err, a, b, Y_test

# Method 2: Original SVD with accounting for
# biases!
def originalSVDwithBellsWhistles(K=10):
    ''' This is the main engine for SVD with accounting
        for biases for each movie and user! '''
    # Make modifications to V from original minimization
    # with changes to V!
    # Calculate original SVD with biases this time.
    U, V, err, a, b, Y_test = naiveMinimization(K=K)
    U, V = centerUV(U, V)

    # SVD of V!
    A, B = SVDofV(V, K=K, max_epochs=300)

    projU, projV = calculateProjection(A, U, V)

    # Look at and compare out of sample errors.
    print(get_err(U.T, V.T, Y_test, a, b))
    print(get_err(projU.T, projV.T, Y_test, a, b))
    return projU, projV


# Original SVD
originalSVD()
print("Finished next")
originalSVDwithBellsWhistles()
