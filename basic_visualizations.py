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

# import original data and clean
path_to_original_movies_file = '../data/movies.txt'
path_to_original_data='../data/data.txt'
movies, duplicate_count, replace_table = \
    cleaner.clean_movies(path_to_original_movies_file, save=True)
data = cleaner.clean_data(replace_table, path_to_original_data, save_new_data='npy')

# # or import cleaned data
# path_to_clean_movies_file = '../data/movies_nodup.txt'
# path_to_clean_data_file = '../data/data_clean.npy'
# movies = cleaner.read_movie_as_dataframe(path_to_clean_movies_file)
# data = np.load(path_to_clean_data_file)

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
# plt.show()
plt.savefig('../plots/basic/all_ratings1.png')

plt.figure() 
labels = ['1-star','2-star','3-star','4-star','5-star'] #定义标签
sizes = count_ratings #每块值
colors = ['red','yellowgreen','lightskyblue','yellow']
explode = (0,0,0,0) 
patches,text1,text2 = plt.pie(sizes,
                      # explode=explode,
                      labels=labels,
                      # colors=colors,
                      autopct = '%3.2f%%', #数值保留固定小数位
                      shadow = True, #无阴影设置
                      startangle =90, #逆时针起始角度设置
                      pctdistance = 0.6) #数值距圆心半径倍数距离

plt.axis('equal')
# plt.show()
plt.savefig('../plots/basic/all_ratings3.png')

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

plt.figure()
plt.hist(id_rating_n[:,1], bins=50, density=True)
plt.axvline(x=average_rating_of_all_movies,linewidth=4, color='r')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(3.2, 1.0, 'Average={:.2f}'.format(average_rating_of_all_movies), 
         fontsize=14, verticalalignment='top', bbox=props)
plt.title('Histogram of All ratings in the MovieLens Dataset')
plt.xlabel('Rating')
# plt.ylabel('Percentage')
# plt.show()
plt.savefig('../plots/basic/all_ratings2.png')

##################################################
#   All ratings of the ten most popular movies   #
##################################################

# popularity = total number of ratings for this movie
movie_popularity = np.zeros([len(movie_rating),2])
movie_popularity[:,0] = movie_rating[:,0]
movie_popularity[:,1] = id_rating_n[:,2]

# sort movies by popularity high to low
movie_popularity = movie_popularity[movie_popularity[:,1].argsort()[::-1]]

# select top 10
tenth_populatiry = movie_popularity[9,1]

# output titles of 10 most popular movies
with open('most_popular.txt','w') as f:
    print('The top 10 most popular movies are:\n',end='')
    print('==============================================\n',end='')
    print('# Ratings\tTitle\n',end='')
    print('----------------------------------------------\n',end='')
    f.write('==============================================\n')
    f.write('The top 10 most popular movies are:\n')
    f.write('----------------------------------------------\n')
    f.write('# Ratings\tTitle\n')
    for i in range(10):
        line = str(int(movie_popularity[i,1])) + '\t\t' + \
            id_title_dict[int(movie_popularity[i,0])] + '\n'
        print(line, end='')
        f.write(line)
    f.write('==============================================\n')
    print('==============================================\n',end='')

# plot histogram of ratings
plt.figure()
plt.hist(movie_popularity[:,1], bins=70)
plt.axvline(x=tenth_populatiry,linewidth=2, color='r')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(100, 400, '10th Popular Movie Has {:} Ratings'.format(int(tenth_populatiry)), 
          fontsize=14, verticalalignment='top', bbox=props)
plt.title('Histogram of Popularity in the MovieLens Dataset')
plt.xlabel('Number of Movies')
plt.ylabel('Number of Ratings')
# plt.show()
plt.savefig('../plots/basic/pop1.png')

# boxplot of all ratings
plt.figure()
plt.boxplot(movie_popularity[:,1])
plt.scatter(np.ones(10),movie_popularity[:10,1],c='r',edgecolor='r')
plt.axhline(y=tenth_populatiry,linewidth=2, color='r')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(1.05, 400, '10th Popular Movie \nHas {:} Ratings'.format(int(tenth_populatiry)), 
          fontsize=14, verticalalignment='top', bbox=props)
plt.title('Boxplot of Popularity in the MovieLens Dataset')
plt.xticks([])
plt.ylabel('Number of Ratings')
# plt.show()
plt.savefig('../plots/basic/pop2.png')

# get an array of ratings of 10 most popular movies [ID, #1, #2, #3, #4, $5]
popular_movie_rating = np.zeros([10,6])
for i in range(10):
    popular_movie_rating[i] = \
        movie_rating[np.where(movie_rating[:,0] == movie_popularity[i,0])]

count_ratings = np.zeros(5)
average_rating_of_popular_movies = 0
for i in range(5):
    count_ratings[i] = np.sum(popular_movie_rating[:,i+1])
    average_rating_of_popular_movies += (i+1) * count_ratings[i]
average_rating_of_popular_movies /= np.sum(popular_movie_rating[:,1:])

plt.figure()
plt.bar([1,2,3,4,5],count_ratings/np.sum(popular_movie_rating[:,1:])*100)
plt.axvline(x=average_rating_of_popular_movies,linewidth=4, color='r')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(2.5, 33, 'Average={:.2f}'.format(average_rating_of_popular_movies), 
         fontsize=14, verticalalignment='top', bbox=props)
plt.title('Histogram of Ratings of 10 Most Popupar Movies')
plt.xlabel('Rating')
plt.ylabel('Percentage')
# plt.show()
plt.savefig('../plots/basic/pop3.png')

avg_rating_for_each_popular_movie = np.zeros(len(popular_movie_rating))
rating_array = np.array([1,2,3,4,5])
for i_movie, movie in enumerate(popular_movie_rating):
    total_ratings_for_this_movie = np.sum(movie[1:])
    if total_ratings_for_this_movie != 0:
        avg_rating_for_each_popular_movie[i_movie] = \
            np.dot(movie[1:],rating_array) / total_ratings_for_this_movie
    else: 
        avg_rating_for_each_popular_movie[i_movie] = 0

# # hist looks ugly...
# plt.figure()
# plt.hist(avg_rating_for_each_popular_movie, bins=50, density=True)
# plt.axvline(x=average_rating_of_popular_movies,linewidth=4, color='r')
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(3.2, 1.0, 'Average={:.2f}'.format(average_rating_of_popular_movies), 
#          fontsize=14, verticalalignment='top', bbox=props)
# plt.title('Histogram of Ratings of 10 Most Popupar Movies')
# plt.xlabel('Rating')
# # plt.ylabel('Percentage')
# plt.show()
# # plt.savefig()
        
# boxplot of popular movie ratings
plt.figure()
plt.boxplot(avg_rating_for_each_popular_movie)
plt.scatter(np.ones(10),avg_rating_for_each_popular_movie,c='r',edgecolor='k')
# plt.axhline(y=tenth_populatiry,linewidth=2, color='r')
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(1.05, 400, '10th Popular Movie \nHas {:} Ratings'.format(int(tenth_populatiry)), 
#           fontsize=14, verticalalignment='top', bbox=props)
plt.title('Boxplot of 10 Most Popular Movies')
plt.xticks([])
plt.ylabel('Average Rating')
# plt.show()
plt.savefig('../plots/basic/pop4.png')


##########################################
#   All ratings of the ten best movies   #
##########################################

# setting a lower threshold on how many ratings the movie go
n_rating_threshold = 10

# sort movies first by number of ratings then by average rating
avg_rating_for_each_movie = id_rating_n
avg_rating_for_each_movie = \
    avg_rating_for_each_movie[avg_rating_for_each_movie[:,2].argsort()[::1]]
avg_rating_for_each_movie = \
    avg_rating_for_each_movie[avg_rating_for_each_movie[:,1].argsort(kind='mergesort')[::-1]]

# plot the rating of first 50 movies and their popularity
plt.figure()
plt.scatter(np.linspace(1,50,50),avg_rating_for_each_movie[:50,1],
            s=avg_rating_for_each_movie[:50,2],edgecolor='k')
plt.title('Average Ratings of Top 50 Movies')
plt.xlabel('Rating Rank')
plt.ylabel('Average Rating')
# plt.show()
plt.savefig('../plots/basic/best1.png')

fig, ax1 = plt.subplots()
plt.title('Average Ratings and Number of Ratings of Top 50 Movies')
color = 'tab:brown'
ax1.set_xlabel('Rating Rank')
ax1.set_ylabel('Average Rating', color=color)
ax1.set_ylim([4,5.1])
ax1.scatter(np.linspace(1,50,50),avg_rating_for_each_movie[:50,1], 
            s=10, edgecolor='k', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Number of Ratings', color=color)  # we already handled the x-label with ax1
ax2.set_ylim([0,2000])
ax2.bar(np.linspace(1,50,50),avg_rating_for_each_movie[:50,2])
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
plt.savefig('../plots/basic/best2.png')

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

# plot the ratings of the best movies and their popularity
plt.figure()
sct = plt.scatter(np.linspace(1,10,10),best_movies[:,1],
            s=best_movies[:,2],edgecolor='k',alpha=0.6)
plt.title('Average Ratings of Top 10 Movies\n(Threshold={:})'.
          format(n_rating_threshold))
plt.xlabel('Rating Rank')
plt.ylabel('Average Rating')
handles, labels = sct.legend_elements(prop="sizes",num=4, color='tab:blue',alpha=0.6)
legend2 = plt.legend(handles, labels, loc="upper right", title="Popularity")
# plt.show()
plt.savefig('../plots/basic/best3.png')

fig, ax1 = plt.subplots()
plt.title('Average Ratings and Number of Ratings of Top 10 Movies\n(Threshold={:})'.
          format(n_rating_threshold))
color = 'tab:brown'
ax1.set_xlabel('Rating Rank')
ax1.set_ylabel('Average Rating', color=color)
ax1.set_ylim([4,4.6])
ax1.scatter(np.linspace(1,10,10),best_movies[:,1], edgecolor='k', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Number of Ratings', color=color)  # we already handled the x-label with ax1
ax2.set_ylim([0,1200])
ax2.bar(np.linspace(1,10,10),best_movies[:,2])
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
plt.savefig('../plots/basic/best4.png')


# output titles of 10 most popular movies
with open('best.txt','w') as f:
    print('The best 10 movies are (number of rating >= {:}):\n'.
          format(n_rating_threshold),end='')
    print('==============================================\n',end='')
    print('Avg Rating\tTitle\n',end='')
    print('----------------------------------------------\n',end='')
    f.write('==============================================\n')
    f.write('The best 10 movies are (number of rating >= {:}):\n'.
            format(n_rating_threshold))
    f.write('----------------------------------------------\n')
    f.write('Avg Rating\tTitle\n')
    for i in range(10):
        line = '{:.3f}\t\t'.format(best_movies[i,1]) + \
            id_title_dict[int(best_movies[i,0])] + '\n'
        print(line, end='')
        f.write(line)
    f.write('==============================================\n')
    print('==============================================\n',end='')




# get an array of ratings of 10 best movies [ID, #1, #2, #3, #4, $5]
best_movie_rating = np.zeros([10,6])
for i in range(10):
    best_movie_rating[i] = \
        movie_rating[np.where(movie_rating[:,0] == best_movies[i,0])]


count_ratings = np.zeros(5)
average_rating_of_best_movies = 0
for i in range(5):
    count_ratings[i] = np.sum(best_movie_rating[:,i+1])
    average_rating_of_best_movies += (i+1) * count_ratings[i]
average_rating_of_best_movies /= np.sum(best_movie_rating[:,1:])

plt.figure()
plt.bar([1,2,3,4,5],count_ratings/np.sum(best_movie_rating[:,1:])*100)
plt.axvline(x=average_rating_of_best_movies,linewidth=4, color='r')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(2.5, 33, 'Average={:.2f}'.format(average_rating_of_best_movies), 
         fontsize=14, verticalalignment='top', bbox=props)
plt.title('Histogram of Ratings of 10 Best Movies\n(Threshold={:})'.
          format(n_rating_threshold))
plt.xlabel('Rating')
plt.ylabel('Percentage')
# plt.show()
plt.savefig('../plots/basic/best5.png')

avg_rating_for_each_best_movie = np.zeros(len(best_movie_rating))
rating_array = np.array([1,2,3,4,5])
for i_movie, movie in enumerate(best_movie_rating):
    total_ratings_for_this_movie = np.sum(movie[1:])
    if total_ratings_for_this_movie != 0:
        avg_rating_for_each_best_movie[i_movie] = \
            np.dot(movie[1:],rating_array) / total_ratings_for_this_movie
    else: 
        avg_rating_for_each_best_movie[i_movie] = 0

# # hist looks ugly...
# plt.figure()
# plt.hist(avg_rating_for_each_best_movie, bins=50, density=True)
# plt.axvline(x=avg_rating_for_each_best_movie,linewidth=4, color='r')
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(3.2, 1.0, 'Average={:.2f}'.format(average_rating_of_popular_movies), 
#           fontsize=14, verticalalignment='top', bbox=props)
# plt.title('Histogram of Ratings of 10 Best Movies')
# plt.xlabel('Rating')
# # plt.ylabel('Percentage')
# plt.show()
# # plt.savefig()
        
# boxplot of ratings of best movies
plt.figure()
plt.boxplot(avg_rating_for_each_best_movie)
plt.scatter(np.ones(10),avg_rating_for_each_best_movie,c='r',edgecolor='k')
plt.title('Boxplot of 10 Best Movies')
plt.xticks([])
plt.ylabel('Average Rating')
# plt.show()
plt.savefig('../plots/basic/best6.png')


# boxplot of best and most popular movies
plt.figure()
plt.boxplot([avg_rating_for_each_best_movie, avg_rating_for_each_popular_movie])
# plt.scatter(np.ones(10),avg_rating_for_each_best_movie,c='r',edgecolor='k',label='Best')
# plt.scatter(np.ones(10)*2,avg_rating_for_each_popular_movie,c='b',edgecolor='b',label='Popular')
plt.title('Boxplot of 10 Best Movies and 10 Most Popular Movies')
plt.xticks(np.arange(1,3), ('Best', 'Most Popular'))
plt.ylabel('Average Rating')
# plt.show()
plt.savefig('../plots/basic/best7.png')


###############################################
#   All ratings of movies from three genres   #
###############################################
# all genres = [Unknown, Action, Adventure, 
#               Animation, Childrens, Comedy, 
#               Crime, Documentary, Drama, 
#               Fantasy, Film-Noir, Horror, 
#               Musical, Mystery, Romance, 
#               Sci-Fi, Thriller, War, Western]
###############################################

genres = ['Comedy','Musical','Horror']
n_bins = [30,10,20]
genre_ratings= []

for i_genre,genre in enumerate(genres):
    
    # create an movie-rating array [movie_id, 1_star, 2_star, 3_star, 4_star, 5_star]
    genre_movies = movies.loc[movies[genre] == 1]
    genre_movie_id_list = genre_movies['Movie Id'].tolist()
    genre_movie_rating = stat.movie_rating_counter(genre_movie_id_list, data)
    
    # calculate average ratings of all movies in this genre
    count_ratings = np.zeros(5)
    average_rating_of_all_movies = 0
    for i in range(5):
        count_ratings[i] = np.sum(genre_movie_rating[:,i+1])
        average_rating_of_all_movies += (i+1) * count_ratings[i]
    average_rating_of_all_movies /= np.sum(genre_movie_rating[:,1:])
    
    plt.figure()
    plt.bar([1,2,3,4,5],count_ratings/np.sum(genre_movie_rating[:,1:])*100)
    plt.axvline(x=average_rating_of_all_movies,linewidth=4, color='r')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(2.5, 33, 'Average={:.2f}'.format(average_rating_of_all_movies), 
             fontsize=14, verticalalignment='top', bbox=props)
    plt.title('Histogram of All Ratings of {:} Movies in the MovieLens Dataset'.
              format(genre))
    plt.xlabel('Rating')
    plt.ylabel('Percentage')
    # plt.show()
    plt.savefig('../plots/basic/genre_'+genre+'.png')
    
    # create an array for [ID, avg_rating, n_ratings]
    id_rating_n = np.zeros([len(genre_movie_rating),3])
    rating_array = np.array([1,2,3,4,5])
    for i_movie, movie in enumerate(genre_movie_rating):
        id_rating_n[i_movie,0] = movie[0]
        total_ratings_for_this_movie = np.sum(movie[1:])
        id_rating_n[i_movie,2] = total_ratings_for_this_movie
        if total_ratings_for_this_movie != 0: # avoid divided by zero error
            id_rating_n[i_movie,1] = \
                np.dot(movie[1:],rating_array) / total_ratings_for_this_movie
    
    plt.figure()
    plt.hist(id_rating_n[:,1], bins=n_bins[i_genre], density=True)
    plt.axvline(x=average_rating_of_all_movies,linewidth=4, color='r')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(3.2, 1.0, 'Average={:.2f}'.format(average_rating_of_all_movies), 
             fontsize=14, verticalalignment='top', bbox=props)
    plt.title('Histogram of All Ratings of {:} Movies in the MovieLens Dataset'.
              format(genre))
    plt.xlabel('Rating')
    # plt.ylabel('Percentage')
    # plt.show()
    plt.savefig(('../plots/basic/genre2_'+genre+'.png'))
    
    genre_ratings.append(id_rating_n[:,1])
    
# boxplot of best and most popular movies
plt.figure()
plt.boxplot([genre_ratings[0], genre_ratings[1], genre_ratings[2]])
plt.title('Boxplot of 3 Genres of Movies')
plt.xticks(np.arange(1,4), (genres[0],genres[1],genres[2]))
plt.ylabel('Average Rating')
# plt.show()
plt.savefig('../plots/basic/genre_box.png')
