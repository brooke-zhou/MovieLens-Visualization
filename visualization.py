import matrixFactorMethods as factor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_cleaning as cleaner
import basic_stats as stat

## import original data and clean
#path_to_original_movies_file = 'data/movies.txt'
#path_to_original_data='data/data.txt'
#movies, duplicate_count, replace_table = cleaner.clean_movies(path_to_original_movies_file, save=True)
#data = cleaner.clean_data(replace_table, path_to_original_data, save_new_data='npy')

# or import cleaned data
path_to_clean_movies_file = 'data/movies_nodup.txt'
path_to_clean_data_file = 'data/data_clean.npy'
movies = cleaner.read_movie_as_dataframe(path_to_clean_movies_file)
data = np.load(path_to_clean_data_file)

# create movie title-ID lookup dictionary
id_title_dict = {}
for index, row in movies.iterrows():
    movie_id = row[0]
    movie_title = row[1]
    id_title_dict[movie_id] = movie_title

# create a list of movie ID's
movie_id_list = movies['Movie Id'].tolist()

# create a movie rating count array [movie_id, 1_star, 2_star, 3_star, 4_star, 5_star]
movie_rating = stat.movie_rating_counter(movie_id_list, data)

###################################################
# VMakes a plot with given data and saves it to fname
###################################################
def make_plot(A1, A2, names, fname, title):
    plt.figure()
    plt.scatter(A1, A2)
    for i, n in enumerate(names):
        plt.annotate(n, (A1[i], A2[i]))
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.title(title)
    plt.savefig(fname)
    plt.show()
    


###################################################
# Visualize the SVD with Bells and Whistles model
###################################################
def visualizeBellsWhistles():
    # Get 2D projection of U and V matrices
    Uproj, Vproj = factor.originalSVDwithBellsWhistles()
    
    # create an array for [ID, avg_rating, n_ratings]
    id_rating_n = np.zeros([len(movie_rating),3])
    rating_array = np.array([1,2,3,4,5])
    for i_movie, movie in enumerate(movie_rating):
        id_rating_n[i_movie,0] = movie[0]
        total_ratings_for_this_movie = np.sum(movie[1:])
        id_rating_n[i_movie,2] = total_ratings_for_this_movie
        if total_ratings_for_this_movie != 0: # avoid divided by zero error
            id_rating_n[i_movie,1] = \
                np.dot(movie[1:],rating_array) / total_ratings_for_this_movie    
    # popularity = total number of ratings for this movie
    movie_popularity = np.zeros([len(movie_rating),2])
    movie_popularity[:,0] = movie_rating[:,0]
    movie_popularity[:,1] = id_rating_n[:,2]
    
    # sort movies by popularity high to low
    movie_popularity = movie_popularity[movie_popularity[:,1].argsort()[::-1]]
    
    ###################
    # Visualize top 10
    ###################
    A1 = []
    A2 = []
    names = []
    for i in range(10):
        names.append(id_title_dict[int(movie_popularity[i,0])])
        proj = Vproj[:,int(movie_popularity[i,0])]
        A1.append(proj[0])
        A2.append(proj[1])
    make_plot(A1, A2, names, '10PopVisualization', 'Visualization of Top 10 Popular Movies')
    
    ###################
    # Visualize best 10
    ###################
    # setting a lower threshold on how many ratings the movie go
    n_rating_threshold = 10
    # sort movies first by number of ratings then by average rating
    avg_rating_for_each_movie = id_rating_n
    avg_rating_for_each_movie = \
        avg_rating_for_each_movie[avg_rating_for_each_movie[:,2].argsort()[::1]]
    avg_rating_for_each_movie = \
        avg_rating_for_each_movie[avg_rating_for_each_movie[:,1].argsort(kind='mergesort')[::-1]]    
    # select the best 10 movies
    best_movies = np.empty((0,3),int)
    n_needed = 10 # need top 10 movies
    n_selected = 0
    for movie in avg_rating_for_each_movie:
        if movie[2] >= n_rating_threshold:
            best_movies = np.append(best_movies, [movie], axis=0)
            n_selected += 1
        if n_selected >= n_needed:
            break
    A1 = []
    A2 = []
    names = []
    for i in range(10):
        names.append(id_title_dict[int(best_movies[i,0])])
        proj = Vproj[:,int(best_movies[i,0])]
        A1.append(proj[0])
        A2.append(proj[1])
    make_plot(A1, A2, names, '10BestVisualization', 'Visualization of Top 10 Best Movies')    
    
    
    
visualizeBellsWhistles()
    
    