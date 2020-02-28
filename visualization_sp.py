# import matrixFactorMethods as factor
# import ImplicitImplementation as implicit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_cleaning as cleaner
import basic_stats as stat

## import original data and clean
#path_to_original_movies_file = 'data/movies.txt'
#path_to_original_data='data/train.txt'
#movies, duplicate_count, replace_table = cleaner.clean_movies(path_to_original_movies_file, save=True)
#data = cleaner.clean_data(replace_table, path_to_original_data, save_new_data='txt')

# or import cleaned data
path_to_clean_movies_file = '../data/movies_nodup.txt'
path_to_clean_data_file = '../data/data_clean.npy'
movies = cleaner.read_movie_as_dataframe(path_to_clean_movies_file)
print(len(movies))
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
# Makes a plot with given data and saves it to fname
###################################################
def make_plot(A1, A2, names, fname, title):
    plt.figure()
    plt.scatter(A1, A2)
    for i, n in enumerate(names):
        plt.annotate(n, (A1[i], A2[i]))
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.title(title)
    # plt.savefig(fname)
    plt.show()
    


#############################
# Visualize the Surpise NMF #
#############################
import SurpriseMatrixFactorization as sp

n_factors=100
# n_epochs=400
n_epochs=2
reg_pu=0.06
reg_qi=0.06
reg_bu=0.02
reg_bi=0.02
lr_bu=0.005
lr_bi=0.005
biased=False
Vproj, Uproj, train_err = sp.surpriseNMF(mode='visualization',
                                n_factors=n_factors, 
                                n_epochs=n_epochs,
                                reg_pu=reg_pu,
                                reg_qi=reg_qi,
                                reg_bu=reg_bu,
                                reg_bi=reg_bi,
                                lr_bu=lr_bu,
                                lr_bi=lr_bi,
                                biased=biased)

# Rescale dimensions to compress the image
for i in range(len(Vproj)):
    Vproj[i] = Vproj[i] / max(Vproj[i])
for i in range(len(Uproj)):
    Uproj[i] = Uproj[i] / max(Uproj[i])
        

print(Vproj.shape)

def visualize_V():
    
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
    make_plot(A1, A2, names, 'out/10PopVisualization', 'Visualization of Top 10 Popular Movies')
    
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
        proj = Vproj[:,(int(best_movies[i,0]))]
        A1.append(proj[0])
        A2.append(proj[1])
    make_plot(A1, A2, names, 'out/10BestVisualization', 'Visualization of Top 10 Best Movies') 
    
    
    # Comedy and Drama plots
    plt.figure()
    comedy_movies = movies.loc[movies["Comedy"] == 1]
    comedy_movie_id_list = comedy_movies['Movie Id'].tolist()
    A1C = []
    A2C = []
    namesC = []
    print(len(comedy_movie_id_list))
    for i in range(len(comedy_movie_id_list)):
        namesC.append(id_title_dict[comedy_movie_id_list[i]])
        proj = Vproj[:,(int(comedy_movie_id_list[i]))]
        A1C.append(proj[0])
        A2C.append(proj[1])  
    plt.scatter(A1C, A2C, c='b')
    #for i, n in enumerate(namesC):
        #plt.annotate(n, (A1C[i], A2C[i]))    
    horror_movies = movies.loc[movies["Horror"] == 1]
    horror_movie_id_list = horror_movies['Movie Id'].tolist()
    A1H = []
    A2H = []
    namesH = []
    for i in range(len(horror_movie_id_list)):
        namesH.append(id_title_dict[horror_movie_id_list[i]])
        proj = Vproj[:,(int(horror_movie_id_list[i]))]
        A1H.append(proj[0])
        A2H.append(proj[1])
    plt.scatter(A1H, A2H, c='r')
    #for i, n in enumerate(namesH):
        #plt.annotate(n, (A1H[i], A2H[i]))    
    musical_movies = movies.loc[movies["Musical"] == 1]
    musical_movie_id_list = musical_movies['Movie Id'].tolist()
    A1M = []
    A2M = []
    namesM = []
    for i in range(len(musical_movie_id_list)):
        namesM.append(id_title_dict[musical_movie_id_list[i]])
        proj = Vproj[:,(int(musical_movie_id_list[i]))]
        A1M.append(proj[0])
        A2M.append(proj[1]) 
    plt.scatter(A1M, A2M, c='g')
    #for i, n in enumerate(namesM):
        #plt.annotate(n, (A1M[i], A2M[i]))
    plt.xlabel('Axis 1')
    plt.ylabel('Axis 2')
    plt.title("Comedy, Horror, and Musical Movies")
    # plt.savefig("out/GenreVis")
    plt.show()
    
    
    
visualize_V()
    
    