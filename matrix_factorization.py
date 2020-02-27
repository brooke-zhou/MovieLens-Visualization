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


# # data cleaning (only need to run once if you store the cleaned files)
# path_to_original_movies_file = '../data/movies.txt'
# path_to_original_data='../data/data.txt'
# movies, duplicate_count, replace_table = \
#     cleaner.clean_movies(path_to_original_movies_file, save=True)
# data = cleaner.clean_data(replace_table, path_to_original_data, save_new_data='txt')
# path_to_train_data='../data/train.txt'
# train_data = cleaner.clean_data(replace_table, path_to_train_data, save_new_data='txt')
# path_to_test_data='../data/test.txt'
# test_data = cleaner.clean_data(replace_table, path_to_test_data, save_new_data='txt')


################
# SVD (biased) #
################

# Step 1: (tune model) get training and test errors (RMSE)
n_factors=100
n_epochs=10
lr_all=0.005
reg_all=0.02
V, U, train_err, test_err = sp.surpriseSVD(mode='evaluation',
                                            n_factors=n_factors, 
                                            n_epochs=n_epochs,
                                            lr_all=lr_all, 
                                            reg_all=reg_all)

# # Step 2: (fit model) use all data to learn U and V and get error (RMSE)
# V, U, train_err = sp.surpriseSVD(mode='visualization',
#                                  n_factors=n_factors, 
#                                  n_epochs=n_epochs,
#                                  lr_all=lr_all, 
#                                  reg_all=reg_all)

# TO DO Step 3: project U, V into a 2D space: (V = A Sigma B)

# TO DO Step 4: use the first two columns of A and visualize the new V 


######################
# PMF (unbiased SVD) #
######################

# # Step 1: (tune model) get training and test errors (RMSE)
# n_factors=100
# n_epochs=10
# lr_all=0.005
# reg_all=0.02
# V, U, train_err, test_err = sp.surprisePMF(mode='evaluation',
#                                            n_factors=n_factors, 
#                                            n_epochs=n_epochs,
#                                            lr_all=lr_all, 
#                                            reg_all=reg_all)

# # Step 2: (fit model) use all data to learn U and V and get error (RMSE)
# V, U, train_err = sp.surprisePMF(mode='visualization',
#                                  n_factors=n_factors, 
#                                  n_epochs=n_epochs,
#                                  lr_all=lr_all, 
#                                  reg_all=reg_all)

# TO DO Step 3: project U, V into a 2D space: (V = A Sigma B)

# TO DO Step 4: use the first two columns of A and visualize the new V 


#########
# SVD++ #
#########

# # Step 1: (tune model) get training and test errors (RMSE)
# n_factors=20
# n_epochs=10
# lr_all=0.007
# reg_all=0.02
# V, U, train_err, test_err = sp.surpriseSVDpp(mode='evaluation',
#                                            n_factors=n_factors, 
#                                            n_epochs=n_epochs,
#                                            lr_all=lr_all, 
#                                            reg_all=reg_all)

# # Step 2: (fit model) use all data to learn U and V and get error (RMSE)
# V, U, train_err = sp.surpriseSVDpp(mode='visualization',
#                                  n_factors=n_factors, 
#                                  n_epochs=n_epochs,
#                                  lr_all=lr_all, 
#                                  reg_all=reg_all)

# TO DO Step 3: project U, V into a 2D space: (V = A Sigma B)

# TO DO Step 4: use the first two columns of A and visualize the new V 


#######
# NMF #
#######

# # Step 1: (tune model) get training and test errors (RMSE)
# n_factors=100
# n_epochs=10
# reg_pu=0.06
# reg_qi=0.06
# reg_bu=0.02
# reg_bi=0.02
# lr_bu=0.005
# lr_bi=0.005
# biased=False
# V, U, train_err, test_err = sp.surpriseNMF(mode='evaluation',
#                                            n_factors=n_factors, 
#                                            n_epochs=n_epochs,
#                                            reg_pu=reg_pu,
#                                            reg_qi=reg_qi,
#                                            reg_bu=reg_bu,
#                                            reg_bi=reg_bi,
#                                            lr_bu=lr_bu,
#                                            lr_bi=lr_bi,
#                                            biased=biased)

# # Step 2: (fit model) use all data to learn U and V and get error (RMSE)
# V, U, train_err = sp.surpriseNMF(mode='visualization',
#                                 n_factors=n_factors, 
#                                 n_epochs=n_epochs,
#                                 reg_pu=reg_pu,
#                                 reg_qi=reg_qi,
#                                 reg_bu=reg_bu,
#                                 reg_bi=reg_bi,
#                                 lr_bu=lr_bu,
#                                 lr_bi=lr_bi,
#                                 biased=biased)

# TO DO Step 3: project U, V into a 2D space: (V = A Sigma B)

# TO DO Step 4: use the first two columns of A and visualize the new V 