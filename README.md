===============================================================================

RIT - ENGL 481/781 - Intro to Natural Language Processing

Finding important features for classification of spam in SMS messages

Doug Dlutz    <djd3681@rit.edu>

Ryan Dennehy  <rmd5947@rit.edu>

===============================================================================

This project involves the extraction of relevant features and the creation
of an effective classifier for the purpose of SMS spam filtering.

The data set used in this project comes from the UC Irvine Machine Learning 
repository, at: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

The Python script should be run as:

    python project.py (training set) (test set)

e.g., with our file naming convention:

    python project.py SMSTrain SMSTest

===============================================================================

The output of the program should be something similar to the following:

    Processing train file
    ('Ham: ', 4827, 'Spam: ', 747)
    Processing test file
    lengths
    838
    6863
    fitting SVM
    {'kernel': 'rbf', 'C': 600, 'verbose': False, 'probability': False, 'degree': 3, 
     'shrinking': True, 'max_iter': -1, 'random_state': None, 'tol': 0.001, 
     'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0005, 'class_weight': None}
    [[728   0]
     [ 10 100]]
    nan     emailAddress
    0.000e+00       phoneNumber
    0.000e+00       numChars
    3.232e-157      Unigram-txt
    6.377e-141      Unigram-call
    2.843e-116      Unigram-free
    1.621e-112      Unigram-claim
    5.359e-87       Unigram-prize
    1.205e-83       Unigram-mobile
    1.193e-75       Unigram-won
    fitting SVM again
    [[725   3]
     [ 22  88]]

===============================================================================
