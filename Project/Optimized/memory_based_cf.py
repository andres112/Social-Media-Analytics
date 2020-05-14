import pandas as pd
import numpy as np


class MeB:

    def __init__(self, train, user_names, movie_names, k_size=5, bounds=(1,5)):
        self.train = train
        self.user_names = user_names
        self.movie_names = movie_names
        self.k = k_size
        self.bounds = bounds

    def pearson_correlation(self):
        self.pearson_corr = pd.DataFrame(self.train.T, columns=self.user_names).corr()
        print(self.pearson_corr.shape)

        # # Implementation from scratch (poor in performance)
        # values = []
        # for user in self.train:
        #     user_values = []
        #     for row in self.train:
        #         user_1 = user
        #         user_2 = row
        #         nans = ~np.logical_or(np.isnan(user_1), np.isnan(user_2))
        #         user_1 = np.compress(nans, user_1)
        #         user_2 = np.compress(nans, user_2)

        #         # Numerator
        #         user1_product = user_1 - user_1.mean()
        #         user2_product = user_2 - user_2.mean()
        #         num = (user1_product*user2_product).sum()

        #         # Denominator
        #         user1_sqr = np.sqrt((user1_product**2).sum())
        #         user2_sqr = np.sqrt((user2_product**2).sum())
        #         den = user1_sqr * user2_sqr

        #         correl = num/den
        #         user_values.append(correl)
        #     values.append(user_values)
        # self.pearson_corr = pd.DataFrame(values, index = self.user_names, columns = self.user_names)

    def neighborhood(self, k):
        qou = int(self.user_names.shape[0])
        neighbors = np.zeros((qou, k)).astype(int)
        item = 0
        for i in self.user_names:
            user_corr = self.pearson_corr[i].drop(
                [i]).sort_values(ascending=False)
            neighbors[item] = user_corr.iloc[:k].index.values
            item = item + 1
        self.neighbors = neighbors
        # print(self.neighbors.shape)

    def get_user_avgs(self):
        user_avg = np.nanmean(self.train, axis=1)
        user_avg = pd.DataFrame(user_avg, index=self.user_names)
        self.user_avg = user_avg

    def predict(self):
        train_df = pd.DataFrame(self.train, index = self.user_names, columns=self.movie_names)
        qom = int(self.movie_names.shape[0])
        qou = int(self.user_names.shape[0])
        item = 0
        predicted_matrix = np.empty((qou, qom))
        predicted_matrix[:] = np.nan
        for user in self.user_names:
            target_avg = self.user_avg.at[user, 0]
            k_neigbors = self.neighbors[item]
            k_correlation = self.pearson_corr.loc[user, k_neigbors].values
            k_ratings = train_df.loc[k_neigbors, :].values
            k_avg = self.user_avg.loc[k_neigbors].values

            # neighbor ratings - neighbor mean
            r_ru = np.subtract(k_ratings, k_avg)
            # (neighbor ratings - neighbor mean) * pearson corr
            r_ru_corr = np.multiply(r_ru.T, k_correlation)
            nan_mask = np.isnan(r_ru_corr)
            r_ru_corr = np.ma.masked_array(r_ru_corr, nan_mask)
            # ((neighbor ratings - neighbor mean) * pearson corr).sumation
            num = np.sum(r_ru_corr, axis=1)

            # similarity.sumation
            den = np.nansum(k_correlation)
            # avoid division by 0
            if(den > 0):
                predicted_matrix[item] = (num / den) + target_avg
            item = item + 1
        predicted_matrix[predicted_matrix < self.bounds[0]] = self.bounds[0]
        predicted_matrix[predicted_matrix > self.bounds[1]] = self.bounds[1]
        return predicted_matrix
