import pandas as pd
import numpy as np
from collections import Counter

class KNN:
    """K Nearest Neighbors algorithm implementation.
    """
    
    def __init__(self, k, problem='classification'):
        """Constructor for KNN class.
        
        Keyword arguments:
        argument k -- number of neighbors
        argument problem -- problem type (classification or regression)
        
        Return: KNN object
        """
        
        self._k = k
        self._problem = problem
        
    def _euclidean_distance(self, x, x_2):
        """Calculate the euclidean distance between two points.
        
        Keyword arguments:
        argument x -- first point
        argument x_2 -- second point
        
        Return: euclidean distance between two points
        """
        
        return np.sqrt(np.sum((x - x_2) ** 2))
    
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.
        
        Keyword arguments:
        argument X -- training data
        argment y -- target values
        
        Return: None
        """
        
        self._X = X
        self._y = y
    
    def predict(self):
        """Predict the class labels for the provided data.
        
        Keyword arguments:
        Return: predicted class labels
        """
        
        return [self._predict_most_common_label(x) for x in self._X]
    
    def _predict_most_common_label(self, x):
        """Predict the class label for the provided data.
        
        Keyword arguments:
        argument x -- data point
        Return: predicted class label
        """
        
        distances = self._get_euclidean_distances(x)
        K_indices = self._get_the_most_common_indices(distances=distances)
        k_nearest_labels = self._get_the_nearest_labels(K_indices)
        
        if self._problem == 'classification':
            return self._get_the_most_common_label(k_nearest_labels)
        elif self._problem == 'regression':
            return self._get_the_mean_of_most_common_labels(k_nearest_labels)
        else:
            raise ValueError('Problem must be classification or regression')

    def _get_euclidean_distances(self, x):
        """Calculate the euclidean distance between a point and all other points.
        
        Keyword arguments:
        argument x -- data point
        Return: euclidean distances between a point and all other points
        """
        
        return [self._euclidean_distance(x, x_train) for x_train in self._X]

    def _get_the_most_common_indices(self, distances):
        """Get the indices of the k nearest neighbors.
        
        Keyword arguments:
        argument distances -- euclidean distances between a point and all other points
        Return: indices of the k nearest neighbors
        """
        
        return np.argsort(distances)[:self._k]
    
    def _get_the_nearest_labels(self, k_indices):
        """Get the labels of the k nearest neighbors.
        
        Keyword arguments:
        argument k_indices -- indices of the k nearest neighbors
        Return: labels of the k nearest neighbors
        """
        
        return [self._y[i] for i in k_indices]
    
    def _get_the_most_common_label(self, k_nearest_labels):
        """Get the most common label of the k nearest neighbors.
        
        Keyword arguments:
        argument k_nearest_labels -- labels of the k nearest neighbors
        Return: most common label of the k nearest neighbors
        """
        
        return Counter(k_nearest_labels).most_common(1)[0][0]
    
    def _get_the_mean_of_most_common_labels(self, k_nearest_labels):
        """Get the mean of the most common labels of the k nearest neighbors.
        
        Keyword arguments:
        argument k_nearest_labels -- labels of the k nearest neighbors
        Return: mean of the most common labels of the k nearest neighbors
        """
        
        return np.mean(k_nearest_labels)