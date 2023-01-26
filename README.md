# MovieLens-Reccomendation-System
![image](https://user-images.githubusercontent.com/117116368/214936782-a38b3e99-382e-45e7-9514-18cf809cc060.png)
## Table of Contents

* [Overview](#Overview)
* [Method](#Process)
* [Analysis](#Exploritory-Data-Analysis)
* [Models](#Model-Implementation-and-Performance)

## Notebooks
* Cleaning.ipynb :Clenaing of dataset
* EDA.ipynb :Exploritory Data Analysis
* CoClustering.ipynb :CoClustering + Tuning
* SVD_Modeling.ipynb :SVD Construction + Tuning
* KNNBaseline.ipynb :KnnBaseline Construction + Tuning
* KNNBasic.ipynb :KnnBasic Construction + Tuning
* SlopeOne.ipynb :SlopeOne Construction + Tuning

## Overview

Recommendation systems are the most succesful application of Machine Learning tech in practice. The goal of this project is to develop a machine learning model that can accurately recommend movies to users based on their past viewing history and preferences. The model will be trained on a dataset of movie ratings provided by the MovieLens website, which contains 58,000 movies by 280,000 users. The model will use collaborative filtering methods to make said recommendations. 

Business Impact: A movie recommendation system can help drive engagement and revenue for a streaming platform or movie rental service. By providing personalized recommendations to users, the platform can increase user satisfaction and retention, and drive additional movie rentals or subscriptions.

Implementation: The project will be implemented in Python, using popular machine learning library Surprise. The model will be trained and tested using the MovieLens dataset, and then implemented on a constructed function to make the top 5 movie recommendation with the highest ratings. 

Expected Outcomes: The model will be able to make accurate recommendations to users based on their past viewing history and preferences, resulting in increased user satisfaction and retention for the platform.

## Data Cleaning

Being the diligent aspiring Data Sceintist first thing was to make sure the dataset I was working with was clean and ready for any EDA, using movies.csv and ratings.csv as they contain all we need for a recommendation system.

* There were a few duplicated movies that conatined the same movieId so I already knew that I had to eliminate those entries.

* Seperating the title object so I can make a new column to include the year that movie was released.

* When merging the ratings and movies csv files they were shadowed by NaN/Null values as there had been a few movies with either no year or recorded rating.

## EDA Analysis

Given I'm focusing on collaborative methods on users, items, and movies I needed to make a file that contained all 3 which can be found in cleaned_movie_ratings.csv

First things first is to see which genres were going to be the most popular given how often they're rated which can be seen below

![Genre_Distribution](https://user-images.githubusercontent.com/117116368/214921144-87948ccb-ee4a-41d5-a056-5309f0db5f51.png)

The most frequent being: Drama, Comedy, Action, Thriller, Adventure, Romance, Sci-fi, and Crime. All of which are well above the mean rate of genre based ratings in the database.

Aside from genres I needed to check out the distribution of ratings as they'll be needed to construct my recommendation model

![Rating_dist](https://user-images.githubusercontent.com/117116368/214923053-240d96b6-f675-451d-8ebd-b585414fae1d.png)

More than 50% of all the ratings are between 3-4, which could pose a problem since as with collaboratve filtering we need to have distributed values to make better predictions based off user ratings.


## Model Implementation and Performance
### SlopeOne

My first model used to establish a benchmark with no tuning or adjustments made was the SlopeOne model as its a simple and efficient model to start off with. It's based on the idea that the difference between the ratings of two items is more important than the actual ratings themselves. The algorithm uses the deviations between the ratings of different items to make predictions for a user.

![Screen Shot 2023-01-26 at 2 10 13 PM](https://user-images.githubusercontent.com/117116368/214927637-8c1e91e1-2e62-4a72-b270-087629a413ea.png)

As seen its not performing all that well, but thats to be expected from a simple model, as its limitations include not predicting well on a diverse movie preference for some users.


### KNNBaseline
My best performing model was the KNNBaseline from the Surprise libaray being of the traditional k-NN algorithm, the prediction for a user's rating on an item is calculated as the average rating of the k-nearest neighbors of the item, where the neighbors are determined based on the similarity between the ratings of the items.

The KNN Baseline algorithm addresses these issues by incorporating a baseline estimate into the prediction process. The baseline estimate is calculated as the average of all the ratings for an item, and it is subtracted from the rating of each user-item pair.

The algorithm then uses the similarity measure and the deviation from the baseline to make predictions for a user's rating on an item. The prediction is calculated as the average of the ratings of the k-nearest neighbors of the item, plus the baseline estimate of the item.

![Screen Shot 2023-01-26 at 2 13 19 PM](https://user-images.githubusercontent.com/117116368/214928276-6f2cc8ec-c23e-4a88-9ff4-283f1da75795.png)

The reason KNNBaseline performed the best could be attributed to the algorithm fitting well to sparse datasets, as it uses the baseline estimate to adjust for the lack of information. However, it's important to note that the algorithm is sensitive to the choice of k and the similarity measure used, which is why I used GridSearchCV to apply several parameters to the algorithm with different cross validation values as well. 


### Final Recommendation 

Once I had my best performing model I constructed a function to pick the top 5 rated movie suggestion for the random user, in this case it was userId[10]. Below is the first 5 rated movies for the user alongside the genres the user usually watches. 

![User10watched](https://user-images.githubusercontent.com/117116368/214931037-1d97f398-ca5f-4286-997d-73813a1ad2dd.png)

After fitting the model and applying UserId to get_n_recommendation function the list below are the highest rated suggestions for the user based off similar interets which is on par with the kind of film the user watches.. just better.

![Screen Shot 2023-01-26 at 12 39 29 PM (2)](https://user-images.githubusercontent.com/117116368/214931663-80742ab8-ce63-4b2c-9a51-f9064aa19a3f.png)


The KNNBasline recommendation model developed for the MovieLens dataset was able to effectively predict user ratings for movies based on their past behavior and preferences. The model employed the use of collaborative filtering techniques, which leveraged the similarities between users and items to make personalized recommendations. The results of the model were evaluated using various metrics such as RMSE and MAE, and it was found to perform well in comparison to other popular recommendation algorithms. Overall, the recommendation model has the potential to improve the movie-watching experience for users by providing them with personalized and relevant movie recommendations.


