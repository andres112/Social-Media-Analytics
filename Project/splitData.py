# Split dataset in train and test data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def split_train_test(data, user_means, fill_with='nan', percent_test=0.1):
    user_means = user_means.to_dict()['movieRating']
    train, test = train_test_split(data, test_size= percent_test)

    train = train.pivot_table(index='movieId', columns="userId", values="movieRating")    
    test = test.pivot_table(index='movieId', columns="userId", values="movieRating")
    
    if(fill_with == 'mean'):
        train = train.fillna(user_means)    
        test = test.fillna(user_means)
    if(fill_with == 'zeros'):
        train = train.fillna(0)    
        test = test.fillna(0)
    
    return train, test
