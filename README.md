# MovieLens-Visualization
Caltech CS155 (2020WI) miniproject 2. Team 0-Loss-or-Bust. This project is to make visualizations of movies in the MovieLens dataset. 

# Prerequisites
- pandas 1.0.0. or higher
- numpy 1.18.1 or higher
- matplotlib 3.1.3 or higher
- Implicit 0.4.0 or higher
- Surprise x.x.x

# Data cleaning
- All related functions are defined in data_cleaning.py
- First, call function clean_movies to detect and remove duplicate movies from movies.txt
- Then, call function clean_data to re-index movies in a given data file

# Basic Visualizations
- All of code used to generate basic visualizations are in basic_visualizations.py
- First, read in clean movie and data files
- Then, run each part of the file to make visualizations of 
  - all movies 
  - most popula movies
  - best movies 
  - movies of selected genres
 
# Matrix Factorization
- Modified code from Homework 5
- Manually implemented biased SVD
- Off-the-shelf implementation: Implicit
- Off-the-shelf implementation: Surprise
  - SurpriseMatrixFactorization.py
   - Functions of four matrix factorization methods (SVD, PMF, SVD++ and NMF) from Surprise to be used for
   - (i) train on training data, then evaluate out-of-sample error on test data
   - (ii) train on all data and output projected U and V in 2 dimensions
  - matrix_factorization.py
   - First, read in clean movie and data files
   - Then, run each part corresponding to each matrix factorization methods to
    - Tune model: get training and test errors (root mean squared error)
    - Train with all data to learn U and V matrices and project them to 2D space

# Matrix Factorization Visualizations
