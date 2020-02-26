#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 06:03:46 2020

@author: root
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_cleaning as cleaner
import SurpriseMatrixFactorization as sp

# # import original data and clean (only need to run once if you store the cleaned files)
# path_to_original_movies_file = '../data/movies.txt'
# path_to_original_data='../data/data.txt'
# movies, duplicate_count, replace_table = \
#     cleaner.clean_movies(path_to_original_movies_file, save=True)
# data = cleaner.clean_data(replace_table, path_to_original_data, save_new_data='txt')
# path_to_train_data='../data/train.txt'
# train_data = cleaner.clean_data(replace_table, path_to_train_data, save_new_data='txt')
# path_to_test_data='../data/test.txt'
# test_data = cleaner.clean_data(replace_table, path_to_test_data, save_new_data='txt')

path_to_test_data='../data/test_clean.txt'
test_data = np.loadtxt(path_to_test_data)

##############
# biased SVD #
##############

# Step 1: learn U and V by Matrix Factorization (Y ~ U^T V)
V, U, _, _ = sp.surpriseSVD(DataPath = '../data/data_clean.txt',
                            n_factors=100, n_epochs=10, 
                            lr_all=0.005, reg_all=0.02)

# Step 2:evaluate test error
test_err = 0
# UT = U.transpose()
for line in test_data:
    user = int(line[0])
    movie = int(line[1])
    Y_test = line[2]
    UTi = U[user-1]
    Vj = V[movie-1]
    Y_pred = np.dot(UTi,Vj)
    test_err += (Y_pred - Y_test)**2


# TO DO Step 3: project U, V into a 2D space: (V = A Sigma B)

# TO DO Step 4: use the first two columns of A and visualize the new V 

# TO DO : repeat step 1-4 for all methods...
