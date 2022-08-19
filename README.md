# Wine Quality Classification
Final project for Machine Learning and Pattern Recognition class at Politecnico di Torino (PoliTO).

## Short Description

This project consists in discriminating between good and bad quality wines using the popular wine dataset from the UCI repository.

The original dataset consists of 10 classes (quality 1 to 10). For the project, the dataset has been binarized, collecting all
wines with low quality (lower than 6) into class 0, and good quality (greater than 6) into class 1. Wines with quality 6 have been discarded to simplify the task

The dataset contains both red and white wines (originally separated, they have been merged).  There are 11 features, that represent physical properties of the
wine. Classes are partially balanced.

## Approach

In this project we want to be Bayesian in the sense that we want to put a prior on the classes and use, as a metric, 
the Detection Cost Function (DCF), also known as **empirical Bayes Risk**, instead of the usual accuracy score:


$$ \mathcal{B}_{emp} = \pi_T C_{fn} P_{fn} +
(1 - \pi_T) C_{fp} P_{fp}
$$

where

* $P_{fn}$ is the false negative rate (false rejection rate)
* $P_{fp}$ is the false positive rate (false acceptance rate)
* $C_{fn}$ is the cost associated to the false negative case
* $C_{fp}$ is the cost associated to the false positive case
* $\pi_T$ is the prior belief associated to the true case


The risk measures the costs of our decisions over the evaluation samples,
taking into account the false positive and negative scenarios according to 
a given cost associated to them.

The metric specifically used is the normalized DCF, i.e.

$$ DCF(\pi_T, C_{fn}, C_{fp}) = \cfrac{B_{emp}}{min (\pi_T C_{fn}, (1-\pi_T)C_{fp})} $$

Hence, we only use models for which we can evaluate a score on the evaluation set.

## Applied Techniques and Models

Since for this class libraries offering already implemented algorithms, like `sklearn`, are not allowed, we also offer an from-scratch
implementation of all the methods used.

In particular:
- Support Vector Classifier with Linear, Polynomial and RBF kernels
- Gaussian Classifier
- Naive Bayes
- Gaussian Mixture Model
- Linear Logistic Regression
- Quadratic Logistic Regression

All of them are applied to the raw data, standardized data and Gaussianized data both for a single train-validation split and for a 5-folds cross validation.

Gaussianized features are obtained computing the inverse of cumulative feature rank $$y = \Phi^{-1} (r(x))$$
being $r(x)$ the rank of a feature over the training set

$$ r(x) = \cfrac{1}{N+2}\left(\sum^{N}_{i=0} X[x < x_i] + 1\right) $$

where $X$ is the data matrix.