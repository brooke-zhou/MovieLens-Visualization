from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import Dataset
from surprise import Reader
from surprise import accuracy

import pandas as pd

#############################
# SVD with bias in Surprise #
#############################
def surpriseSVD(mode,
                DataPath = '../data/data_clean.txt',
                TrainPath = '../data/train_clean.txt',
                TestPath = '../data/test_clean.txt',
                n_factors=100, 
                n_epochs=20, 
                lr_all=0.005, 
                reg_all=0.02, 
                verbose=True):
    
    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))
    
    if mode == 'evaluation':
        
        # train data processing
        train = pd.read_csv(TrainPath, sep="\t", header=None)
        train.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(train[["User Id", "Movie Id", "Rating"]], reader)
        trainset = data.build_full_trainset()
        
        # fit model
        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, biased=True, 
            init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
            verbose=verbose)
        algo.fit(trainset)
        
        # evaluate train error
        test = trainset.build_testset()
        predictions = algo.test(test)
        train_err = accuracy.rmse(predictions, verbose=False)
        
        # test data processing
        test = pd.read_csv(TestPath, sep="\t", header=None)
        test.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(test[["User Id", "Movie Id", "Rating"]], reader)
        testset = data.build_full_trainset()
        
        # evaluate train error
        test = testset.build_testset()
        predictions = algo.test(test)
        test_err = accuracy.rmse(predictions, verbose=False)
    
        # Return V (qi),  U (pu), train_err (RMSE), test_err (RMSE)
        return algo.qi, algo.pu, train_err, test_err
    
    elif mode == 'visualization':
        
        # train data processing
        alldata = pd.read_csv(DataPath, sep="\t", header=None)
        alldata.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(alldata[["User Id", "Movie Id", "Rating"]], reader)
        trainset = data.build_full_trainset()
        
        # fit model
        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, biased=True, 
            init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
            verbose=verbose)
        algo.fit(trainset)
        
        # evaluate train error
        test = trainset.build_testset()
        predictions = algo.test(test)
        train_err = accuracy.rmse(predictions, verbose=False)
        
        # Return V (qi),  U (pu), train_err (RMSE)
        return algo.qi, algo.pu, train_err
        

################################################################
# probabilistic factorization (PMF) (unbiased SVD) in Surprise #
################################################################
def surprisePMF(mode,
                DataPath = '../data/data_clean.txt',
                TrainPath = '../data/train_clean.txt',
                TestPath = '../data/test_clean.txt',
                n_factors=100, 
                n_epochs=20, 
                lr_all=0.005, 
                reg_all=0.02, 
                verbose=True):
    
    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))
    
    if mode == 'evaluation':
        
        # train data processing
        train = pd.read_csv(TrainPath, sep="\t", header=None)
        train.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(train[["User Id", "Movie Id", "Rating"]], reader)
        trainset = data.build_full_trainset()
        
        # fit model
        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, biased=False, 
            init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
            verbose=verbose)
        algo.fit(trainset)
        
        # evaluate train error
        test = trainset.build_testset()
        predictions = algo.test(test)
        train_err = accuracy.rmse(predictions, verbose=False)
        
        # test data processing
        test = pd.read_csv(TestPath, sep="\t", header=None)
        test.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(test[["User Id", "Movie Id", "Rating"]], reader)
        testset = data.build_full_trainset()
        
        # evaluate train error
        test = testset.build_testset()
        predictions = algo.test(test)
        test_err = accuracy.rmse(predictions, verbose=False)
    
        # Return V (qi),  U (pu), train_err (RMSE), test_err (RMSE)
        return algo.qi, algo.pu, train_err, test_err
    
    elif mode == 'visualization':
        
        # train data processing
        alldata = pd.read_csv(DataPath, sep="\t", header=None)
        alldata.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(alldata[["User Id", "Movie Id", "Rating"]], reader)
        trainset = data.build_full_trainset()
        
        # fit model
        algo = SVD(n_factors=n_factors, n_epochs=n_epochs, biased=False, 
            init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
            verbose=verbose)
        algo.fit(trainset)
        
        # evaluate train error
        test = trainset.build_testset()
        predictions = algo.test(test)
        train_err = accuracy.rmse(predictions, verbose=False)
        
        # Return V (qi),  U (pu), train_err (RMSE)
        return algo.qi, algo.pu, train_err
        
    
#####################################################################
# An extension of SVD taking into account implicit ratings (SVD++) #
#####################################################################
def surpriseSVDpp(mode,
                DataPath = '../data/data_clean.txt',
                TrainPath = '../data/train_clean.txt',
                TestPath = '../data/test_clean.txt',
                n_factors=20, 
                n_epochs=20, 
                lr_all=0.007, 
                reg_all=0.02, 
                verbose=True):
    
    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))
    
    if mode == 'evaluation':
        
        # train data processing
        train = pd.read_csv(TrainPath, sep="\t", header=None)
        train.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(train[["User Id", "Movie Id", "Rating"]], reader)
        trainset = data.build_full_trainset()
        
        # fit model
        algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, 
            init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
            verbose=verbose)
        algo.fit(trainset)
        
        # evaluate train error
        test = trainset.build_testset()
        predictions = algo.test(test)
        train_err = accuracy.rmse(predictions, verbose=False)
        
        # test data processing
        test = pd.read_csv(TestPath, sep="\t", header=None)
        test.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(test[["User Id", "Movie Id", "Rating"]], reader)
        testset = data.build_full_trainset()
        
        # evaluate train error
        test = testset.build_testset()
        predictions = algo.test(test)
        test_err = accuracy.rmse(predictions, verbose=False)
    
        # Return V (qi),  U (pu), train_err (RMSE), test_err (RMSE)
        return algo.qi, algo.pu, train_err, test_err
    
    elif mode == 'visualization':
        
        # train data processing
        alldata = pd.read_csv(DataPath, sep="\t", header=None)
        alldata.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(alldata[["User Id", "Movie Id", "Rating"]], reader)
        trainset = data.build_full_trainset()
        
        # fit model
        algo = SVDpp(n_factors=n_factors, n_epochs=n_epochs, 
            init_mean=0, init_std_dev=0.1, lr_all=lr_all, reg_all=reg_all,
            verbose=verbose)
        algo.fit(trainset)
        
        # evaluate train error
        test = trainset.build_testset()
        predictions = algo.test(test)
        train_err = accuracy.rmse(predictions, verbose=False)
        
        # Return V (qi),  U (pu), train_err (RMSE)
        return algo.qi, algo.pu, train_err
    

####################################
# Non-negative factorization (NMF) #
####################################
def surpriseNMF(mode,
                DataPath = '../data/data_clean.txt',
                TrainPath = '../data/train_clean.txt',
                TestPath = '../data/test_clean.txt',
                n_factors=15, 
                n_epochs=50,
                reg_pu=0.06,
                reg_qi=0.06,
                reg_bu=0.02,
                reg_bi=0.02,
                lr_bu=0.005,
                lr_bi=0.005,
                init_low=0,
                init_high=1,
                biased=False,
                verbose=True):
    
    # We need the rating scale.
    reader = Reader(rating_scale=(1, 5))
    
    if mode == 'evaluation':
        
        # train data processing
        train = pd.read_csv(TrainPath, sep="\t", header=None)
        train.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(train[["User Id", "Movie Id", "Rating"]], reader)
        trainset = data.build_full_trainset()
        
        # fit model
        algo = NMF(n_factors=n_factors, 
                    n_epochs=n_epochs,
                    reg_pu=reg_pu,
                    reg_qi=reg_qi,
                    reg_bu=reg_bu,
                    reg_bi=reg_bi,
                    lr_bu=lr_bu,
                    lr_bi=lr_bi,
                    init_low=init_low,
                    init_high=init_high,
                    biased=biased,
                    verbose=verbose)
        algo.fit(trainset)
        
        # evaluate train error
        test = trainset.build_testset()
        predictions = algo.test(test)
        train_err = accuracy.rmse(predictions, verbose=False)
        
        # test data processing
        test = pd.read_csv(TestPath, sep="\t", header=None)
        test.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(test[["User Id", "Movie Id", "Rating"]], reader)
        testset = data.build_full_trainset()
        
        # evaluate train error
        test = testset.build_testset()
        predictions = algo.test(test)
        test_err = accuracy.rmse(predictions, verbose=False)
    
        # Return V (qi),  U (pu), train_err (RMSE), test_err (RMSE)
        return algo.qi, algo.pu, train_err, test_err
    
    elif mode == 'visualization':
        
        # train data processing
        alldata = pd.read_csv(DataPath, sep="\t", header=None)
        alldata.columns = ["User Id", "Movie Id", "Rating"]
        data = Dataset.load_from_df(alldata[["User Id", "Movie Id", "Rating"]], reader)
        trainset = data.build_full_trainset()
        
        # fit model
        algo = NMF(n_factors=n_factors, 
                    n_epochs=n_epochs,
                    reg_pu=reg_pu,
                    reg_qi=reg_qi,
                    reg_bu=reg_bu,
                    reg_bi=reg_bi,
                    lr_bu=lr_bu,
                    lr_bi=lr_bi,
                    init_low=init_low,
                    init_high=init_high,
                    biased=biased,
                    verbose=verbose)
        algo.fit(trainset)
        
        # evaluate train error
        test = trainset.build_testset()
        predictions = algo.test(test)
        train_err = accuracy.rmse(predictions, verbose=False)
        
        # Return V (qi),  U (pu), train_err (RMSE)
        return algo.qi, algo.pu, train_err
    
    