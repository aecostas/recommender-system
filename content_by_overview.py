import pandas as pd
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

metadata = pd.read_csv('./data/movies_metadata.csv', low_memory=False)

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

# Construct the required TF-IDF matrix by fitting and transforming the data
# TF-IDF = Term frequency – Inverse document frequency
tfidf_matrix = tfidf.fit_transform(q_movies['overview'])

# linear_kernel: http://scikit-learn.org/stable/modules/metrics.html#linear-kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(q_movies.index, index=q_movies['title']).drop_duplicates()

recommendations = get_recommendations('The Dark Knight Rises', indices, q_movies, cosine_sim)

print(vars(recommendations))
