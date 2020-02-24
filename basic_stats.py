#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 22:54:13 2020

@author: yacong.zhou@gmail.com
"""

import numpy as np

def movie_rating_counter(movie_id_list, data_array):
    """
    Count number of different ratings for each movie.

    Parameters
    ----------
    movie_id_list : list
        DESCRIPTION.
    data_array : numpy array
        (N, 1) array of movie ID's where N = number of (non-duplicate) movies.

    Returns
    -------
    movie_rating : numpy array
        (N, 6) array. Col_1 = movie_id, Col_2-6 are the count of 1-5 ratings.

    """
    
    # initialize an array to store movie rating counts
    n_movie = len(movie_id_list)
    movie_rating = np.expand_dims(movie_id_list, axis=1)
    movie_rating = np.concatenate((movie_rating, np.zeros([n_movie,5])), axis=-1)
    
    # create a look-up dictionary to accelerate
    id_index = {}
    indices = np.linspace(0, n_movie-1, n_movie)
    for movie_id, index in zip(movie_id_list, indices):
        id_index[movie_id] = int(index)
        
    # loop over all data entries
    for row in data_array:
        movie_id = int(row[1])
        rating = int(row[2])
        movie_rating[id_index[movie_id],rating] += 1
        
    return movie_rating