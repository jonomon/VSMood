import matplotlib.pyplot as plt
import numpy as np

from vsbrnn.utils import get_log_likelihood

class MultiInstance():
    def __init__(self, method, X_train, y_train, x_test, trainer):
        self.method = method
        self.X_train = X_train
        self.y_train = y_train
        self.x_test = x_test
        self.trainer = trainer

    def get_pred(self, preds):
        if self.method == "mean":
            return self.get_mean_pred(preds)
        elif self.method == "1std-mean":
            return self.get_n_std_mean_pred(1, preds)
        elif self.method == "max-likelihood":
            return self.get_max_likelihood(preds)
        elif self.method == "similar":
            return self.get_similar(preds)
        elif self.method == "log-prob":
            return self.get_log_prob(preds)
        else:
            return None

    def get_mean_pred(self, preds):
        return np.mean(preds)

    def get_n_std_mean_pred(self, n, preds):
        std = np.std(preds)
        mean = np.mean(preds)
        max_value = mean + n * std
        min_value = mean - n * std
        mean_preds = preds[np.logical_and(preds > min_value, preds < max_value)]
        return np.mean(mean_preds)

    def get_max_likelihood(self, preds):
        X_predicts = self.trainer.predict(self.X_train)
        n_d, bins_d, _ = plt.hist(
            X_predicts[self.y_train[:, 1]==1], facecolor='green', alpha=0.5)
        n_bd, bins_bd, _ = plt.hist(
            X_predicts[self.y_train[:, 1]==0], facecolor='red', alpha=0.5)

        log_like = [get_log_likelihood(a, n_bd, bins_bd,
                                       n_d, bins_d) for a in preds]
        return np.mean(log_like)

    def get_similar(self, preds):
        sequences = self.x_test["seq"]
        n = sequences.shape[0]
        distances = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sequence1 = sequences[i, :]
                    sequence2 = sequences[j, :]
                    leven_dist = self.levenshteinDistance(sequence1, sequence2)
                    distances[i, j] = leven_dist
        
        mean_distances = np.mean(distances, axis=1)
        max_distance_index = np.argmax(mean_distances)
        preds_max_removed = np.delete(preds, max_distance_index)
        return np.mean(preds_max_removed)
                    
    def levenshteinDistance(self, s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def get_log_prob(self, preds):
        log_preds = np.log(preds)
        log_preds = np.clip(log_preds, -1e15, 1e15)
        return np.mean(log_preds)
