import implicit
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from statistics import mean


def implicitModel(movieLensDataTrainPath='../data/train_clean.txt', movieLensDataTestPath='../data/test_clean.txt'):
    dfTrain = pd.read_csv(movieLensDataTrainPath, sep="\t", header=None)
    dfTrain.columns = ["User Id", "Movie Id", "Rating"]

    dfTest = pd.read_csv(movieLensDataTestPath, sep="\t", header=None)
    dfTest.columns = ["User Id", "Movie Id", "Rating"]

    test = dfTest.to_numpy()
    train = dfTrain.to_numpy()

    # initialize a model
    model = implicit.als.AlternatingLeastSquares(factors=25, iterations=400, regularization=0.01)
    # print(train)
    M = max(max(train[:, 0]), max(test[:, 0])).astype(int)
    N = max(max(train[:, 1]), max(test[:, 1])).astype(int)

    newTrains = np.zeros((M, N))
    # print(len(newTrains))
    # print(len(newTrains[0]))
    for y in train:
        i, j, yij = y
        i = i - 1
        j = j - 1
        # print(newTrains[i])
        newTrains[i][j] = yij
    newTrains = np.array(newTrains)
    train = csr_matrix(newTrains)
    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(train)

    # print(model.item_factors)
    # print(len(model.item_factors))
    # print(model.user_factors)
    # print(len(model.user_factors))
    return model.item_factors, model.user_factors
    # recommend items for a user
    # user_items = item_user_data.T.tocsr()
    # recommendations = model.recommend(userid, user_items)

    # find related items
    # related = model.similar_items(itemid)

def get_err2(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    totalLength = len(Y)

    sumOfSqs = 0
    meanYs = mean(Y[:, 2])
    for y in Y:
        #print(y)
        i = int(y[0])
        j = int(y[1])
        yij = y[2]
        i = i - 1
        j = j - 1
        sumOfSqs = sumOfSqs + ((yij - meanYs - np.dot(U[i], V[j])) ** 2)

    normSum = (np.linalg.norm(U, ord='fro') ** 2 + np.linalg.norm(V, ord='fro') ** 2)
    return ((reg * normSum) + sumOfSqs) / (2 * totalLength)


def Vtrain(M, N, K, eta, reg, Y, max_epochs=300):
    model = implicit.als.AlternatingLeastSquares(factors=25, iterations=400, regularization=0.01)
    # print(train)

    newTrains = np.array(Y)
    train = csr_matrix(newTrains)
    print(train.shape)
    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(train)

    print(model.item_factors.shape)
    # print(len(model.item_factors))
    print(model.user_factors.shape)
    # print(len(model.user_factors))
    return model.item_factors, model.user_factors

def SVDofV(oldV):
    M = len(oldV)  # users
    N = len(oldV[0])  # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    # Ks = [10, 20, 30, 50, 100]
    #print("oldV")
    #print(oldV)
    K = 20
    reg = 0.0
    eta = 0.03  # learning rate
    # Use to compute Ein and Eout
    A, B = Vtrain(M, N, K, eta, reg, oldV, max_epochs=300)
    #print(err)
    return A, B


def tryThis():
    U, V = implicitModel()
    U = np.float64(U)
    V = np.float64(V)
    U = U.T
    V = V.T
    # U = np.array(U)
    # V = np.array(V)
    for i in range(len(V)):
        V[i] = V[i] - mean(V[i])
    for i in range(len(U)):
        U[i] = U[i] - mean(U[i])
    # SVD of V!

    A, B = SVDofV(V)
    A = A.T
    # Use the first 2 cols for work
    Asub = A[:, :2]

    projU = np.dot(Asub.T, U)
    projV = np.dot(Asub.T, V)

    # Rescale dimensions to compress the image
    for i in range(len(projV)):
        projV[i] = projV[i] / max(projV[i])
    for i in range(len(projU)):
        projU[i] = projU[i] / max(projU[i])
    dfTest = pd.read_csv('../data/test_clean.txt', sep="\t", header=None)
    dfTest.columns = ["User Id", "Movie Id", "Rating"]

    Y_test = dfTest.to_numpy()
    print(get_err2(U.T, V.T, Y_test))
    print(get_err2(projU.T, projV.T, Y_test))
    return projU, projV

tryThis()