# K-Nearest Neighbors (KNN)

## Main idea

KNN (K-Nearest Neighbors) is a machine learning algorithm that can be used for both classification and regression problems. The main idea behind KNN is to find the K closest data points to a new data point and use the majority class (in classification) or the average value (in regression) of those K data points to predict the class or value of the new data point.

## Use cases

KNN can be used in a variety of problems, such as image recognition, text classification, and recommendation systems. However, it may not be suitable for large datasets, as it requires calculating the distance between the new data point and all other data points in the dataset. Additionally, KNN assumes that all features have the same importance, which may not always be the case.

## Advantages and disavantages

One advantage of KNN is that it is easy to understand and implement. It also does not require training, as it simply stores the entire dataset. 

However, it can be slow and memory-intensive, especially for large datasets. It also may not perform well if the dataset has a lot of noise or irrelevant features.

## Points of attention

When using KNN, it is important to choose the right value of K, as a small value of K may result in overfitting, while a large value of K may result in underfitting, you can use hyper parameter tuning to find the best value of k. It is also important to normalize the features to ensure that they are on the same scale.

## References

[How to find the optimal value of k in KNN](https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb)

[KNN (K-Nearest Neighbors)](https://towardsdatascience.com/knn-k-nearest-neighbors-1-a4707b24bd1d)