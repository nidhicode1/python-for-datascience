import pandas as pd
import numpy as np
import text_normalizer as tn
# import model_evaluation_utils as meu  ## Not provided to you
# The Intern can code the methods to print the Metric like accuracy,
# and confusion matrix. The o/ps in the below cells would give an idea.

np.set_printoptions(precision=2, linewidth=80)



dataset = pd.read_csv(r'movie_reviews.csv')

# take a peek at the data
print(dataset.head())
reviews = np.array(dataset['review'])
sentiments = np.array(dataset['sentiment'])

# build train and test datasets
train_reviews = reviews[:35000]
train_sentiments = sentiments[:35000]
test_reviews = reviews[35000:]
test_sentiments = sentiments[35000:]

# normalize datasets
norm_train_reviews = tn.normalize_corpus(train_reviews)
norm_test_reviews = tn.normalize_corpus(test_reviews)

# Traditional Supervised Machine Learning Models

## Feature Engineering

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# build BOW features on train reviews
cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0, ngram_range=(1,2))
cv_train_features = cv.fit_transform(norm_train_reviews)

# build TFIDF features on train reviews
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1,2),
                     sublinear_tf=True)
tv_train_features = tv.fit_transform(norm_train_reviews)


# transform test reviews into features
cv_test_features = cv.transform(norm_test_reviews)
tv_test_features = tv.transform(norm_test_reviews)

print('BOW model:> Train features shape:', cv_train_features.shape, ' Test features shape:', cv_test_features.shape)
print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)

## Model Training, Prediction and Performance Evaluation
from sklearn.linear_model import SGDClassifier, LogisticRegression

lr = LogisticRegression(penalty='l2', max_iter=100, C=1)
svm = SGDClassifier(loss='hinge', max_iter=100)

# Logistic Regression model on BOW features
# Please Note : the module meu is not been provided.
lr_bow_predictions = meu.train_predict_model(classifier=lr,
                                             train_features=cv_train_features, train_labels=train_sentiments,
                                             test_features=cv_test_features, test_labels=test_sentiments)
meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=lr_bow_predictions,
                                      classes=['positive', 'negative'])

# THE BELOW O/P SHOULD GIVE YOU A FAIR IDEA ON WHAT :
# methods like
# train_predict_model() are doing and printing as o/p.
# display_model_performance_metrics() are doing and printing as o/p.

# As an Intern you are not suppose to produce the exact o/p
# You may only code the minimum required metrics which helps you to
# compare the different ML models.

# Logistic Regression model on TF-IDF features
# Please Note : the module meu is not been provided.
lr_tfidf_predictions = meu.train_predict_model(classifier=lr,
                                               train_features=tv_train_features, train_labels=train_sentiments,
                                               test_features=tv_test_features, test_labels=test_sentiments)
meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=lr_tfidf_predictions,
                                      classes=['positive', 'negative'])

# THE BELOW O/P SHOULD GIVE YOU A FAIR IDEA ON WHAT :
# methods like
# train_predict_model() are doing and printing as o/p.
# display_model_performance_metrics() are doing and printing as o/p.

# As an Intern you are not suppose to produce the exact o/p
# You may only code the minimum required metrics which helps you to
# compare the different ML models.

# SVM model on BOW features
# Please Note : the module meu is not been provided.
svm_bow_predictions = meu.train_predict_model(classifier=svm,
                                             train_features=cv_train_features, train_labels=train_sentiments,
                                             test_features=cv_test_features, test_labels=test_sentiments)
meu.display_model_performance_metrics(true_labels=test_sentiments,
                                      predicted_labels=svm_bow_predictions,
                                      classes=['positive', 'negative'])

# THE BELOW O/P SHOULD GIVE YOU A FAIR IDEA ON WHAT :
# methods like
# train_predict_model() are doing and printing as o/p.
# display_model_performance_metrics() are doing and printing as o/p.

# As an Intern you are not suppose to produce the exact o/p
# You may only code the minimum required metrics which helps you to
# compare the different ML models.

# SVM model on TF-IDF features
# Please Note : the module meu is not been provided.
svm_tfidf_predictions = meu.train_predict_model(classifier=svm,
                                                train_features=tv_train_features, train_labels=train_sentiments,
                                                test_features=tv_test_features, test_labels=test_sentiments)
meu.display_model_performance_metrics(true_labels=test_sentiments,
                                      predicted_labels=svm_tfidf_predictions,
                                      classes=['positive', 'negative'])

# THE BELOW O/P SHOULD GIVE YOU A FAIR IDEA ON WHAT :
# methods like
# train_predict_model() are doing and printing as o/p.
# display_model_performance_metrics() are doing and printing as o/p.

# As an Intern you are not suppose to produce the exact o/p
# You may only code the minimum required metrics which helps you to
# compare the different ML models.