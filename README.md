# Movie-Ratings-Predictor
Problem description:

In this we will use the movie ratings collected from everyone
in the class to build a ratings predictor.

Training set: Use the data in train.csv to train the recommendation/ratings system. It is
the same as the data collected as part of Movie Night survey with 1) the email
addresses blanked out, and IDs converted to consecutive numbers. 2) The ratings of
the last four movies removed.

Test set: The data in test.csv will be the test set. You are to predict the ratings of
the movie Despicable Me . The recommendation system should output <ID, Rating
[Despicable Me]> pair for all the records in the test set. A sample submission file --
sampleSubmission.csv is provided. The test data is gathered from surveying people
who are enrolled in other courses.
Ideas for fine-tuning the ratings predictor
1) Employee neighborhood-based collaborative filtering techniques:
user-based, or item-based.
2) Latent factor models (matrix factorization).
3) Combination of methods: you will build more than one
recommender, and combine the outputs of the different recommenders using say
a linear regression model. The linear regression model itself may be trained
using the training dataset.
