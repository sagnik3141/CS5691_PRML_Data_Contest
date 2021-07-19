# Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import catboost

# Loading Data

train = pd.read_csv('train.csv')
songs = pd.read_csv('songs.csv')
song_labels = pd.read_csv('song_labels.csv')
test = pd.read_csv('test.csv')
save_for_later = pd.read_csv('save_for_later.csv')

from sklearn.model_selection import train_test_split
train_init, train_rem = train_test_split(train, train_size = 0.3, random_state = 1023)

customer_id_list = train['customer_id'].unique()
song_id_list = train['song_id'].unique()

# Initializing Matrix Factorization
learning_rate = 1e-2
iters = 1
dim = 100
reg = 0.05
# Initialization

customer_weights = {}
for customer in customer_id_list:
    np.random.seed(0)
    customer_weights[customer] = np.random.uniform(0, 10e-10, dim)
    
song_weights = {}
for song in song_id_list:
    np.random.seed(0)
    song_weights[song] = np.random.uniform(0, 10e-10, dim)
print('Matrix Fact started')    
# Training Matrix Factorization
for i in range(iters):
    for k in range(len(train_init.index.to_numpy())):
        customer_weight = customer_weights[train_init['customer_id'].iloc[k]]
        song_weight = song_weights[train_init['song_id'].iloc[k]]
        y = train_init['score'].iloc[k]
        
        temp = y - np.dot(customer_weight, song_weight)
        customer_weight_new = customer_weight + learning_rate*(temp*song_weight-reg*customer_weight)
        song_weight_new = song_weight + learning_rate*(temp*customer_weight-reg*song_weight)
        customer_weights[train_init['customer_id'].iloc[k]] = customer_weight_new
        song_weights[train_init['song_id'].iloc[k]] = song_weight_new
print('mat fact ended')        
customer_weights_df = pd.DataFrame(customer_weights)
customer_weights_df = customer_weights_df.transpose()
customer_weights_df['customer_id'] = customer_weights_df.index
train_rem = train_rem.merge(customer_weights_df, on = 'customer_id', how = 'left')

song_weights_df = pd.DataFrame(song_weights)
song_weights_df = song_weights_df.transpose()
song_weights_df['song_id'] = song_weights_df.index
songs = songs.merge(song_weights_df, on = 'song_id', how = 'left')

estimates_train = []
for k in range(len(train_rem.index.to_numpy())):
    customer_weight = customer_weights[train_rem['customer_id'].iloc[k]]
    song_weight = song_weights[train_rem['song_id'].iloc[k]]
    estimate = np.dot(customer_weight, song_weight)
    estimates_train.append(estimate)
    
train_rem['estimates'] = estimates_train

song_labels_pivot = song_labels.pivot_table(index = 'platform_id', columns = 'label_id', values = 'count')
song_labels_pivot = song_labels_pivot.fillna(0)
song_labels_pivot = song_labels_pivot.applymap(lambda x: np.log(1+np.abs(x)))
print('nmf start')
from sklearn.decomposition import NMF
nmf = NMF(n_components=100, max_iter = 10, random_state = 1, verbose = 1)

song_labels_transformed = nmf.fit_transform(song_labels_pivot)
song_labels_transformed_df = pd.DataFrame(song_labels_transformed, index = song_labels_pivot.index)
songs = pd.merge(songs, song_labels_transformed_df, on = 'platform_id', how = 'left')
songs = songs.drop(['platform_id'], axis = 1)
print('nmf end')
train_song_mean = train.groupby('song_id').mean()
song_scores = train.merge(train_song_mean, on = 'song_id', how = 'left')
song_scores = song_scores[['song_id', 'score_y']]
song_scores.drop_duplicates('song_id', keep = 'first', inplace = True)
songs = songs.merge(song_scores, on = 'song_id', how = 'left')

song_num_ratings = train['song_id'].value_counts().to_frame()
song_num_ratings['num_ratings'] = song_num_ratings['song_id']
song_num_ratings['song_id'] = song_num_ratings.index
songs = songs.merge(song_num_ratings, on = 'song_id', how = 'left')

songs.drop_duplicates('song_id', keep = 'first', inplace = True)

f = pd.merge(train_rem, save_for_later, on=['customer_id','song_id'], how='left', indicator='Exist')
train_rem = f

X_train = pd.merge(train_rem, songs, on = ['song_id'], how = 'left')
Y_train = X_train['score']
X_train.drop(['score'], axis = 1, inplace = True)
X_train['released_year'] = X_train['released_year'].fillna(-999)
X_train['language'] = X_train['language'].fillna('none')
X_train['number_of_comments'] = X_train['number_of_comments'].fillna(-999)

from catboost import CatBoostRegressor

model = CatBoostRegressor(depth = 8, num_trees = 100, random_seed = 100)

model.fit(X_train, Y_train, cat_features = [0,1, 103, 105])
print('Train Finished')
test = test.merge(customer_weights_df, on = 'customer_id', how = 'left')

estimates_test = []
for k in range(len(test.index.to_numpy())):
    customer_weight = customer_weights[test['customer_id'].iloc[k]]
    song_weight = song_weights[test['song_id'].iloc[k]]
    estimate = np.dot(customer_weight, song_weight)
    estimates_test.append(estimate)
    
test['estimates'] = estimates_test
print('Estimated')
test = pd.merge(test, save_for_later, on=['customer_id','song_id'], how='left', indicator='Exist')

X_test = pd.merge(test, songs, on = ['song_id'], how = 'left')
X_test['released_year'] = X_test['released_year'].fillna(-999)
X_test['language'] = X_test['language'].fillna('none')
X_test['number_of_comments'] = X_test['number_of_comments'].fillna(-999)
print('predstart')
y_test_pred = model.predict(X_test)
print('predend')
y_final = pd.DataFrame(y_test_pred)
y_final['score'] = y_final[0]
y_final.drop(0, axis = 1, inplace = True)
y_final['test_row_id'] = y_final.index
y_final = y_final[['test_row_id', 'score']]
print('final ready')
y_final.to_csv('predicted.csv', index = False)
print('executed')