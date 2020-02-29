# MovieLens-Visualization
Caltech CS155 (2020WI) miniproject 2. Team 0-Loss-or-Bust
This project is to make visualizations of movies in the MovieLens dataset. 

# Prerequisites
pandas 1.0.0. or higher
numpy 1.18.1 or higher
matplotlib 3.1.3 or higher
implicit 0.4.0 or higher
surprise x.x.x

# Data cleaning
All related functions are defined in data_cleaning.py
First, call function clean_movies to detect and remove duplicate movies from movies.txt
Then, call function clean_data to re-index movies in a given data file

# Basic Visualizations
All of code used to generate basic visualizations are in basic_visualizations.py
First, read in clean movie and data files
Then, run each part of the file to make visualizations of 
 - all movies 
 - most popula movies
 - best movies 
 - movies of selected genres
 
# Matrix Factorization
1. Modified code from Homework 5
2. Manually implemented biased SVD
3. Off-the-shelf implementation: Implicit
4. Off-the-shelf implementation: Surprise
 - SurpriseMatrixFactorization.py
   This file has four matrix factorization methods (SVD, PMF, SVD++ and NMF) from Surprise and can be use to 
   (i) train on training data, then evaluate out-of-sample error on test data
   (ii) train on all data and output projected U and V in 2 dimensions
 - matrix_factorization.py
   

# Matrix Factorization Visualizations
