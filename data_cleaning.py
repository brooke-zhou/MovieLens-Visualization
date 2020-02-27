#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:27:26 2020

@author: yacong.zhou@gmail.com
"""

import numpy as np
import pandas as pd

def read_movie_as_dataframe(path_to_movies_file='../data/movies.txt'):
    """
    Read movie file as pandas DataFrame

    Parameters
    ----------
    path_to_movies_file : str, optional
        Path to the movie list file. The default is '../data/movies.txt'.

    Returns
    -------
    movie_data : Pandas DataFrame
        Movie data table with proper header.

    """
    movie_data = pd.read_csv(path_to_movies_file, 
                         sep="\t", 
                         names=['Movie Id', 
                                'Movie Title', 
                                'Unknown', 
                                'Action', 
                                'Adventure', 
                                'Animation', 
                                'Childrens', 
                                'Comedy', 
                                'Crime', 
                                'Documentary',
                                'Drama', 
                                'Fantasy', 
                                'Film-Noir', 
                                'Horror', 
                                'Musical', 
                                'Mystery', 
                                'Romance', 
                                'Sci-Fi', 
                                'Thriller', 
                                'War', 
                                'Western'],
                         header=None)
    return movie_data


def clean_movies(path_to_movies_file='../data/movies.txt', save=True):
    """
    Read list of movies, detect and remove duplicates, then re-index all 
    movies from 0 to N_non_dup.

    Parameters
    ----------
    path_to_movies_file : str, optional
        Path to the movie list file. The default is '../data/movies.txt'.
    save : Boolean, optional
        Option to save a cleaned movie file as original format.

    Returns
    -------
    non_duplicate_movies : Pandas DataFrame
        DataFrame of movies with duplicate rows removed and re-indexed.
        Duplicates are dropped except for the first occurrence.
    duplicate_count : dict
        Key = (str) movie title. Value: (int) number of appearance in original list.
    replace_table : dict
        Key = (int) movie id of duplicates. Value = (int) movie id to replace with.

    """

    # read movie data from file as pandas dataframe
    movie_data = read_movie_as_dataframe(path_to_movies_file)
    
    # detect duplicates and sort  
    duplicate_movies = movie_data[movie_data.duplicated(['Movie Title'],keep=False)]
    duplicate_movies = duplicate_movies.sort_values(by=['Movie Title','Movie Id'])
    
    # store duplicate ID and titles
    duplicate_count = {}
    dup_replace_table = {}
    for index, row in duplicate_movies.iterrows():
        movie_id = row[0]
        movie_title = row[1]
        if movie_title not in duplicate_count:
            duplicate_count[movie_title] = 1
            replacement = movie_id
        else:
            duplicate_count[movie_title] += 1
            dup_replace_table[movie_id] = replacement
        
    # store a new dataframe of movies w/o duplicates
    non_duplicate_movies = movie_data.drop_duplicates(['Movie Title'],
                                                      keep='first',
                                                      ignore_index=True)
    # create a dict where keys are old movie IDs and values are re-indexed IDs
    replace_table = {}
    for movie_index in range(len(non_duplicate_movies)):
        replace_table[non_duplicate_movies.iloc[movie_index,0]] = movie_index
        non_duplicate_movies.at[movie_index, 'Movie Id'] = movie_index
    
    # save the new dataframe
    if save:
        non_duplicate_movies.to_csv(path_to_movies_file[:-4]+'_nodup.txt',
                                    sep='\t', header=False, index=False)
    
    # return things
    return non_duplicate_movies, duplicate_count, replace_table


def clean_data(replace_table, path_to_data='../data/data.txt', save_new_data='npy'):
    """
    Replace duplicated movie IDs according to replace_table in user-movie-rating data.

    Parameters
    ----------
    replace_table : dict 
        Key = (int) movie id of duplicates. Value = (int) movie id to replace with.
    path_to_data : str, optional
        Path to data file. The default is '../data/data.txt'.
    save_new_data : str or False, optional
        Option for saving the cleaned data. 
        'npy' : save as numpy array. [User Id, Movie Id, Rating] (default) 
        'txt' : save as text. [User Id, Movie Id, Rating]
        False : do not save the cleaned data

    Returns
    -------
    data : numpy array
         [User Id, Movie Id, Rating] data where dupliciate movie IDs are replaced.

    """
    
    # read data from file as numpy arrray (User Id, Movie Id, Rating)
    data = np.loadtxt(path_to_data)
    
    # replace duplicated movies IDs according to replace_table
    for index,line in enumerate(data):
        if line[1] in replace_table:
            data[index,1] = replace_table[line[1]]
            
    # save new data
    new_filename = path_to_data[:-4]
    if save_new_data == 'npy':
        np.save(new_filename+'_clean.npy', data)
    elif save_new_data == 'txt':
        np.savetxt(new_filename+'_clean.txt', data, fmt='%i\t%i\t%i')
    elif save_new_data == False:
        pass
    else:
        print('Unrecognized save_new_data option.')
        
    return data

