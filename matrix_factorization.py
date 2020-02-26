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

# import original data and clean
path_to_original_movies_file = '../data/movies.txt'
path_to_original_data='../data/data.txt'
movies, duplicate_count, replace_table = \
    cleaner.clean_movies(path_to_original_movies_file, save=True)
data = cleaner.clean_data(replace_table, path_to_original_data, save_new_data='txt')

##############
# biased SVD #
##############

# Step 1: learn U and V by Matrix Factorization (Y ~ U^T V)
V, U = sp.surpriseSVD(DataPath = '../data/data_clean.txt')

# TO DO Step 2:evaluate test error

# TO DO Step 3: project U, V into a 2D space: (V = A Sigma B)

# TO DO Step 4: use the first two columns of A and visualize the new V 

# TO DO : repeat step 1-4 for all methods...
