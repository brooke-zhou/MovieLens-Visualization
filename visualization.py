import matrixFactorMethods as factor
import ImplicitImplementation as implicit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SurpriseMatrixFactorization as sp

import data_cleaning as cleaner
import basic_stats as stat


######################################################################
# Import in data                                                     #
# Shouldn't have to change anything here other than maybe file paths #
######################################################################

## import original data and clean
#path_to_original_movies_file = 'data/movies.txt'
#path_to_original_data='data/train.txt'
#movies, duplicate_count, replace_table = cleaner.clean_movies(path_to_original_movies_file, save=True)
#data = cleaner.clean_data(replace_table, path_to_original_data, save_new_data='txt')
# or import cleaned data
path_to_clean_movies_file = 'data/movies_nodup.txt'
path_to_clean_data_file = 'data/data_clean.npy'
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


################################
# Helper function to make plot #
################################
def make_plot(A1, A2, names, fname, title):
    plt.figure()
    plt.scatter(A1, A2, c='r')
    for i, n in enumerate(names):
        plt.annotate(n, (A1[i], A2[i]))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(title)
    plt.savefig(fname)
    plt.show()
    


#####################
# Visualize the SVD #
#####################
def visualize():
    
    ############################################
    # Get 2D projection of U and V matrices    #
    # Just change f_end, t_end, and V/U source #
    #########################################@@#
    # Used to generate file name and plot titles 
    f_end = "SVD"
    t_end = " (SVD Without Bias)"
    #Uproj, Vproj = factor.originalSVDwithBellsWhistles()
    #Uproj, Vproj = implicit.tryThis()
    Uproj, Vproj = factor.originalSVD()
    # # NMF
    n_factors=100
    n_epochs=400
    reg_pu=0.06
    reg_qi=0.06
    reg_bu=0.02
    reg_bi=0.02
    lr_bu=0.005
    lr_bi=0.005
    biased=False
    #Vproj, Uproj, train_err = sp.surpriseNMF(mode='visualization',
    #                                n_factors=n_factors, 
    #                                n_epochs=n_epochs,
    #                                reg_pu=reg_pu,
    #                                reg_qi=reg_qi,
    #                                reg_bu=reg_bu,
    #                                reg_bi=reg_bi,
    #                                lr_bu=lr_bu,
    #                                lr_bi=lr_bi,
    #                                biased=biased)    
    print(Vproj.shape)
    
    ##########################
    # Set up maps for movies #
    ########################3#
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
    
    ####################
    # Visualize top 10 #
    ####################
    A1 = []
    A2 = []
    names = []
    for i in range(10): # Already sorted
        names.append(id_title_dict[int(movie_popularity[i,0])])
        proj = Vproj[:,int(movie_popularity[i,0])]
        A1.append(proj[0])
        A2.append(proj[1])
    make_plot(A1, A2, names, 'out/Popular10' + f_end, 'Visualization of Top 10 Popular Movies' + t_end)
    
    ##############################
    # Visualize top 10 in genres #
    ##############################
    genres = ["Comedy", "Musical", "Horror"]
    for genre in genres:
        genre_movies = movies.loc[movies[genre] == 1]
        genre_movie_id_list = genre_movies['Movie Id'].tolist()           
        idxs = []
        found = 0
        for idx, pop in movie_popularity:
            if idx in genre_movie_id_list:
                idxs.append(idx)
                found += 1
            if found >= 10:
                break
        A1 = []
        A2 = []
        names = []
        for idx in idxs:
            names.append(id_title_dict[int(idx)])
            proj = Vproj[:,int(idx)]
            A1.append(proj[0])
            A2.append(proj[1])
        make_plot(A1, A2, names, 'out/Popular10' + genre + f_end, 'Visualization of Top 10 Popular ' + genre + ' Movies' + t_end)
        
    ##########################
    # Visualize Hitchcock 10 #
    ##########################
    ids = [184, 476, 598, 475, 439, 486, 485, 610, 500, 602]
    A1 = []
    A2 = []
    names = []
    for idx in ids:
        names.append(id_title_dict[int(idx)])
        proj = Vproj[:,int(idx)]
        A1.append(proj[0])
        A2.append(proj[1])
    make_plot(A1, A2, names, 'out/Hitchcock10' + f_end, 'Visualization of 10 Hitchcock Movies' + t_end)    
    
    #################################
    # Visualize Light and Dark 10   #
    # More obvious when not working #
    #################################
    L1 = []
    L2 = []
    namesL = []
    idsL = [34, 62, 77, 101, 113]
    for idx in idsL:
        namesL.append(id_title_dict[idx])
        proj = Vproj[:,int(idx)]
        L1.append(proj[0])
        L2.append(proj[1])  
    counter = 1
    D1 = []
    D2 = []
    namesD = []
    tagsD = []
    idsD = [55, 95, 126, 178, 194]
    for idx in idsD:
        namesD.append(id_title_dict[idx])
        proj = Vproj[:,int(idx)]
        D1.append(proj[0])
        D2.append(proj[1]) 
    plt.figure()
    plt.scatter(L1, L2, c='c', label='Light-hearted Movies')
    plt.scatter(D1, D2, c='r', label='Dark Movies')
    print(namesL)
    print(namesD)
    for i, n in enumerate(namesL):
        plt.annotate(n, (L1[i], L2[i]))
        #plt.annotate(counter, (L1[i], L2[i])) # use for idx labeling
        counter += 1
    for i, n in enumerate(namesD):
        plt.annotate(n, (D1[i], D2[i]))    
        #plt.annotate(counter, (D1[i], D2[i])) # use for idx labeling
        counter += 1        
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.title("Light-hearted and Dark Movies" + t_end)
    plt.savefig("out/Selected10" + f_end)
    plt.show()
        
    #####################
    # Visualize best 10 #
    #####################
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
    make_plot(A1, A2, names, 'out/Best10' + f_end, 'Visualization of Top 10 Best Movies' + t_end) 
    
    
visualize()
    
    