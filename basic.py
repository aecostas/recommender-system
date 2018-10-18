# https://www.datacamp.com/community/tutorials/recommender-systems-python
# Weighted rating
# v = number of votes
# m = minimum of votes to be listed
# C = mean vote across the whole report

# Import Pandas
import pandas as pd

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Load Movies Metadata
metadata = pd.read_csv('./data/movies_metadata.csv', low_memory=False)

# C is the mean vote across the whole report
C = metadata['vote_average'].mean()

# Calculate the minimum number of votes required to be in the chart, m
quantile = 0.75
m = metadata['vote_count'].quantile(quantile)
print("m (at quantile: %f): %d" % (quantile, m))

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

q_movies['score'] = q_movies.apply(weighted_rating, args=(m, C), axis=1)

q_movies = q_movies.sort_values('score', ascending=False)

print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))
