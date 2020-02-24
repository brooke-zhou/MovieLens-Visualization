#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:32:32 2020

@author: yacong.zhou@gmail.com, xxxxxxxxx@xx.xx, xxxxxxxxx@xx.xx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_cleaning as cleaner
import basic_stats as stat


###########################
#   Read and clean data   #
###########################

# # import original data and clean
# path_to_original_movies_file = '../data/movies.txt'
# path_to_original_data='../data/data.txt'
# movies, duplicate_count, replace_table = \
#     cleaner.clean_movies(path_to_original_movies_file, save=True)
# data = cleaner.clean_data(replace_table, path_to_original_data, save_new_data='npy')

# or import cleaned data
path_to_clean_movies_file = '../data/movies_nodup.txt'
path_to_clean_data_file = '../data/data_clean.npy'
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

# create a movie rating count list
movie_rating = stat.movie_rating_counter(movie_id_list, data)


############################################
#   All ratings in the MovieLens Dataset   #
############################################

count_ratings = np.zeros(5)
average_rating_of_all_movies = 0
for i in range(5):
    count_ratings[i] = np.sum(movie_rating[:,i+1])
    average_rating_of_all_movies += (i+1) * count_ratings[i]
average_rating_of_all_movies /= np.sum(movie_rating[:,1:])

plt.figure()
plt.bar([1,2,3,4,5],count_ratings/np.sum(movie_rating[:,1:])*100)
plt.axvline(x=average_rating_of_all_movies,linewidth=4, color='r')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(2.5, 33, 'Average={:.2f}'.format(average_rating_of_all_movies), 
         fontsize=14, verticalalignment='top', bbox=props)
plt.title('Histogram of All ratings in the MovieLens Dataset')
plt.xlabel('Rating')
plt.ylabel('Percentage')
plt.show()
# plt.savefig()

avg_rating_for_each_movie = np.zeros(len(movie_rating))
rating_array = np.array([1,2,3,4,5])
for i_movie, movie in enumerate(movie_rating):
    total_ratings_for_this_movie = np.sum(movie[1:])
    if total_ratings_for_this_movie != 0:
        avg_rating_for_each_movie[i_movie] = \
            np.dot(movie[1:],rating_array) / total_ratings_for_this_movie
    else: 
        avg_rating_for_each_movie[i_movie] = 0

plt.figure()
plt.hist(avg_rating_for_each_movie, bins=50, density=True)
plt.axvline(x=average_rating_of_all_movies,linewidth=4, color='r')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(3.2, 1.0, 'Average={:.2f}'.format(average_rating_of_all_movies), 
         fontsize=14, verticalalignment='top', bbox=props)
plt.title('Histogram of All ratings in the MovieLens Dataset')
plt.xlabel('Rating')
# plt.ylabel('Percentage')
plt.show()
# plt.savefig()

##################################################
#   All ratings of the ten most popular movies   #
##################################################

# popularity = total number of ratings for this movie
movie_popularity = np.zeros([len(movie_rating),2])
movie_popularity[:,0] = movie_rating[:,0]
for i_movie, movie in enumerate(movie_rating):
    movie_popularity[i_movie,1] = np.sum(movie[1:])

# plt.figure()
plt.hist(movie_popularity[:,1], bins=70)
# plt.axvline(x=average_rating_of_all_movies,linewidth=4, color='r')
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(3.2, 1.0, 'Average={:.2f}'.format(average_rating_of_all_movies), 
#          fontsize=14, verticalalignment='top', bbox=props)
# plt.title('Histogram of All ratings in the MovieLens Dataset')
# plt.xlabel('Rating')
# # plt.ylabel('Percentage')
# plt.show()
# plt.savefig()

# TO DO : select top 10 and analysis

##########################################
#   All ratings of the ten best movies   #
##########################################
# TO DO

###############################################
#   All ratings of movies from three genres   #
###############################################
# TO DO
