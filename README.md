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
- All code used to generate basic visualizations are in basic_visualizations.py
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
    - Functions of four matrix factorization methods (SVD, PMF, SVD++ and NMF) from Surprise to 
      - train on training data, then evaluate out-of-sample error on test data
      - train on all data and output projected U and V in 2 dimensions
  - matrix_factorization.py
    - First, read in clean movie and data files
    - Then, run each part corresponding to each matrix factorization methods to
      - tune model: get training and test errors (root mean squared error)
      - train with all data to learn U and V matrices and project them to 2D space
  - matrixFactorMethods.py
    - Miscellaneous
      - Lines 18-46 have to do with centering and projecting our U and V matrices like we’re supposed         to in the Miniproject guide. These contain the functions centerUV() and calculateProjection().
      - Lines 436-469 contain the function surpriseSVD(), which was the original implementation of surpriseSVD we based our final implementation on.
    - HW 5 SVD implementation
      - Lines 103-120 include the SVDofV() which similarly has to do with projecting U and V by calculating the SVD of V. 
      - grad_U1(), grad_V1() (our gradient functions), get_err2() (RMSE error function), Vtrain_model(),  and originalSVD(), contained in lines 48-232, compute the SVD from homework 5 and create the 2D projections of U and V. 
    - Manual Bias SVD Implementation 
Lines 232-432 contain the functions grad_U(), grad_V(), grad_a(), grad_b() (gradient function), get_err() (RMSE function), train_model(), naiveMinimization(), and originalSVDwithBellsWhistles(), which altogether compute the SVD from homework 5 with the additional conditions of handling user/movie biases and producing the 2D projections at the end.
  - implicitImplementation.py
     - Lines 9-45 includes the implementation for implicitModel, which initializes and runs the model on our datasets. 
    - Vtrain() in lines 74-94 with SVDofV() creates and runs the model specifically for our V to create its SVD. 
    - tryThis() is the main engine from lines 97-121 which calls the functions to create our Us and V’s, center them, create the SVD of V, calculate the 2D projection of U and V, retrieve the RMSEs. The projected U and V are returned at the end.


# Matrix Factorization Visualizations
