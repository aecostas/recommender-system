import pandas as pd
import sys
from ast import literal_eval
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, indices, movies, cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

metadata = pd.read_csv('./data/movies_metadata.csv', low_memory=False)
credits = pd.read_csv('./data/credits.csv')
keywords = pd.read_csv('./data/keywords.csv')

# Remove rows with bad IDs.
metadata = metadata.drop([19730, 29503, 35587])

# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Calculate the minimum number of votes required to be in the chart, m
# en este caso hacemos esto por capacidad de computo. En un ordenador personas
# no es capaz de procesar la matrix completa
quantile = 0.75
m = metadata['vote_count'].quantile(quantile)

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

# Replace NaN with an empty string
q_movies['overview'] = q_movies['overview'].fillna('')

# I need to add this line because q_movies had the
# original index, what will cause an exception in get_recommendation
q_movies = q_movies.reset_index(drop=True)


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    q_movies[feature] = q_movies[feature].apply(literal_eval)

q_movies['director'] = q_movies['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    q_movies[feature] = q_movies[feature].apply(get_list)

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    q_movies[feature] = q_movies[feature].apply(clean_data)

q_movies['soup'] = q_movies.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(q_movies['soup'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)
q_movies = q_movies.reset_index()
indices = pd.Series(q_movies.index, index=q_movies['title'])

recommendations = get_recommendations('The Dark Knight Rises', indices, q_movies, cosine_sim)

print(recommendations)

for key, value in recommendations.items():
	movie = metadata[metadata['id'] == key]
	print(key, movie)
	print('==============')
