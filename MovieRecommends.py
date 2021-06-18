import pandas as pd
def movie_recommender(movie_name):
    df=pd.read_csv(r"C:\Machine Learning\Datasets\MovieLens Dataset\movie.csv")
    genres_dict={'Angry':['Horror','Action','Thriller','IMAX'],'Happy':['Adventure','Animation','Children','Comedy','Fantasy',],'Surprise':['Adventure','Fanatsy','Sci-Fi','Horror'],'Neutral':['Adventure','Comedy','Romance','Drama','Western','Thriller'],'Sad':['Comedy','Romance','Action']}
    frames=[]
    for i in genres_dict['Angry']:
        df2=df[df['genres'].str.contains(i)]
        frames.append(df2)
    result=pd.concat(frames)
    rating=pd.read_csv(r"C:\Machine Learning\Datasets\MovieLens Dataset\rating.csv")
    del rating['timestamp']
    final_df=pd.merge(result,rating,on='movieId',how='left')
    df=final_df
    rating['num of ratings'] = pd.DataFrame(df.groupby('movieId')['rating'].count())
    combine_movie_rating = df.dropna(axis = 0, subset = ['title'])
    movie_ratingCount = (combine_movie_rating.
         groupby(by = ['title'])['rating'].
         count().
         reset_index().
         rename(columns = {'rating': 'totalRatingCount'})
         [['title', 'totalRatingCount']]
        )
    movie_ratingCount.head()
    rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
    rating_with_totalRatingCount.head()
    popularity_threshold = 50
    rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    movie_features_df=rating_popular_movie.pivot_table(index='title',columns='userId',values='rating').fillna(0)
    from scipy.sparse import csr_matrix
    movie_features_df_matrix = csr_matrix(movie_features_df.values)
    from sklearn.neighbors import NearestNeighbors
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(movie_features_df_matrix)
    import numpy as np
    query_index=movie_name
    distances, indices = model_knn.kneighbors(movie_features_df.loc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
    for i in range(0, len(distances.flatten())):
        if i == query_index:
            print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
        else:
            if(distances.flatten()[i]==0.0):
                pass
            else:
                print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))

movie_name="Princess Bride, The (1987)"
movie_recommender(movie_name)