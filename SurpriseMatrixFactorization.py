from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

import pandas as pd

#############################
# SVD with bias in Surprise #
#############################
def surpriseSVD(DataPath = '../data/data_clean.txt', 
    n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=True):
    ''' Basic use of the surprise SVD algorithm. '''
    ''' Params: movieLensDataPath is the path to the movielens data we're looking at. '''
    ''' Note: replace with cleaned data. '''
    ''' We want to return U and V where for a Y of a matrix of movie ratings, Y ~/= U^TV.'''
    
    # Load the data as a pandas data frame, as reading from text didn't quite work at first.
    df = pd.read_csv(DataPath, sep="\t", header=None)
    df.columns = ["User Id", "Movie Id", "Rating"]

    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))

    # The columns are User Id, Movie Id, and Rating.
    data = Dataset.load_from_df(df[["User Id", "Movie Id", "Rating"]], reader)

    # Use the famous SVD algorithm. To fit, we have to convert it to a trainset.
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, biased=True, 
        init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
        verbose=verbose)
    algo.fit(trainset)

    # # Return V (qi),  U (pu), bias of U (bu) and bias of V (bi)
    # return algo.qi, algo.pu, algo.bu, algo.bi
    # Return V (qi),  U (pu)
    return algo.qi, algo.pu


################################################################
# probabilistic factorization (PMF) (unbiased SVD) in Surprise #
################################################################
def surprisePMF(DataPath = '../data/data_clean.txt', 
    n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=True):
    ''' Basic use of the surprise SVD algorithm. '''
    ''' Params: movieLensDataPath is the path to the movielens data we're looking at. '''
    ''' Note: replace with cleaned data. '''
    ''' We want to return U and V where for a Y of a matrix of movie ratings, Y ~/= U^TV.'''
    
    # Load the data as a pandas data frame, as reading from text didn't quite work at first.
    df = pd.read_csv(DataPath, sep="\t", header=None)
    df.columns = ["User Id", "Movie Id", "Rating"]

    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))

    # The columns are User Id, Movie Id, and Rating.
    data = Dataset.load_from_df(df[["User Id", "Movie Id", "Rating"]], reader)

    # Use the famous SVD algorithm. To fit, we have to convert it to a trainset.
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, biased=False, 
        init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
        verbose=verbose)
    algo.fit(trainset)

    # Return U (qi) and  V (pu)
    return algo.qi, algo.pu

#####################################################################
# An extension of SVD taking into account implicit ratings (SVD++) #
#####################################################################
def surpriseSVDpp(DataPath = '../data/data_clean.txt', 
    n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=True):
    ''' Basic use of the surprise SVD algorithm. '''
    ''' Params: movieLensDataPath is the path to the movielens data we're looking at. '''
    ''' Note: replace with cleaned data. '''
    ''' We want to return U and V where for a Y of a matrix of movie ratings, Y ~/= U^TV.'''
    
    # Load the data as a pandas data frame, as reading from text didn't quite work at first.
    df = pd.read_csv(DataPath, sep="\t", header=None)
    df.columns = ["User Id", "Movie Id", "Rating"]

    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))

    # The columns are User Id, Movie Id, and Rating.
    data = Dataset.load_from_df(df[["User Id", "Movie Id", "Rating"]], reader)

    # Use the famous SVD algorithm. To fit, we have to convert it to a trainset.
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs,
        init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
        verbose=verbose)
    algo.fit(trainset)

    # Return U (qi) V (pu), biases of U and V and the (implicit) item factors
    # return algo.qi, algo.pu, algo.bu, algo.bi, algo.yj
    return algo.qi, algo.pu


####################################
# Non-negative factorization (NMF) #
####################################
def surpriseNMF(DataPath = '../data/data_clean.txt', biased=False,
    n_factors=15, n_epochs=50, lr_all=0.005, reg_all=0.02, verbose=True):
    ''' Basic use of the surprise SVD algorithm. '''
    ''' Params: movieLensDataPath is the path to the movielens data we're looking at. '''
    ''' Note: replace with cleaned data. '''
    ''' We want to return U and V where for a Y of a matrix of movie ratings, Y ~/= U^TV.'''
    
    # Load the data as a pandas data frame, as reading from text didn't quite work at first.
    df = pd.read_csv(DataPath, sep="\t", header=None)
    df.columns = ["User Id", "Movie Id", "Rating"]

    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))

    # The columns are User Id, Movie Id, and Rating.
    data = Dataset.load_from_df(df[["User Id", "Movie Id", "Rating"]], reader)

    # Use the famous SVD algorithm. To fit, we have to convert it to a trainset.
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs,
        init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
        verbose=verbose)
    algo.fit(trainset)

    # Return U (qi) V (pu)
    return algo.qi, algo.pu
