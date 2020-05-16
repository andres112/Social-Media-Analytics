# Project 5 - Social Recommendation Systems (CiaoDVD Dataset)
The aim of this project is to build a tool that implements and compares different types of recommendation algorithms on a real-world dataset. The technical aspects are described here below.

## Index
* [Dataset Description](#dataset)
* [Code Testing](#testing)
* Run different recommendation algorithms on the dataset 
* Compare and discuss the results, parameter sensitivity

# <a name="dataset"></a> Dataset Description
first the dataset description and link

libraries used, numpy, pandas, ipython, matplotlib, jupyterlab

files structure
Explanation why there are py and ipynb versions

Pearson correlation settings
- dla.genre_user_rating_analysis (False) for no ploting the graphs
- prunning data m,u,r
- spliting dataset 80% 20%
- Calculate the Pearson correlation of the training dataset (Pearson correlation image)
- k size variation
- Rating prediction calculation with weighted average rating (image here)
- RMSE and MAE
- uncomment external files storing code

PMF:
- train_validate_test_split_pmf in 60% Trainig, 20% Validation Data, 20% Testing Data
- Create number Rating matrix with shape (# of users, # movies)
- Parameters settings
    - lambda_u = 0.02
lambda_v = 0.02
latent_dims = (5, 50)
learn_rate = 0.005
num_iters = 1000
bounds = (1, 5)
- PMF model (image)
- U, V = pmf_model.train(train_data=train_data, validation_data=validation_data) (image)
- with model already trained test the prediction for test predictions = pmf_model.predict(data=test_data)
- RMSE and MAE
- results stored in results/results_pmf1

# <a name="testing"></a> Code Testing
testing:

Before run the code is required to set the parameters for prunning the dataset, although by default this parameters are set to run considering the full dataset values(this process can take huge time, mainly in the pearson correlation section). If you want to reduce the dataset size is possible in 3 ways:
* setting a min value for number of ratings by user. This is posible changing the variable 'p_user' to the value and how = 'u'
* setting a min value for number of ratings by movie. This is posible changing the variable 'p_movie' to the value and how = 'u'
* setting a number to prune randomly. This is posible seting variable p_rnd with a number greater than 0 and less or equal than 1 and how = 'r'

The correlation matrix is computed for all users in matrix once, if the train matrix previously obtained doesnt change. If this change or the dataset change, is required recompute the Pearson correlation matrix. This step is required because the target user should identify their most similar users. 

The testing is by default coded with a loop to make multiple executions, but the functions could be used to compute the neigborhood, and predictions just giving the k size

PMF
