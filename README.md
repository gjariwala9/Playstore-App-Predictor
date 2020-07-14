# Play Store App Rating and Review's Sentiment Predictor: Project Overview 
* Created a tool that estimates rating (MAE ~ 0.31) to get an idea how the app will do on playstore.
* Also, created another tool to clasify the sentiment of the playstore reviews.
* Engineered features from the size of the app to seprate the the app size in Mb and Kb and the size which can varies. 
* Optimized Linear, Lasso, Xgboost and Random Forest Regressors using GridsearchCV and RandomSearchCV to reach the best model. 
* Built a client facing API using flask 

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** pandas, numpy, sklearn, NLTK, tensorflow, matplotlib, seaborn, flask, pickle, joblib  
**For Web Framework Requirements:**  ```pip install -r requirements.txt``` 
**Dataset:** https://www.kaggle.com/lava18/google-play-store-apps 

## Data Cleaning and Preprocessing
After downloading the data, I needed to clean it up so that it was usable for my model. I made the following changes and created the following variables:

**For Rating Prediction**

*	Handled NULL values in the dataset
*	Made columns for Size in Mb, Size in Kb and Size that can varies 
*	Converted the price into float values 
*	Seprate out the genres list 
*	Grouped Android versions
*	Added a column for last updated year

**For Review Sentiment Classification**

*	Handled NULL values in the dataset
*	Removed stopwords from the reviews
*	Did Stemming on the reviews
*	Performed CountVectorizer on the reviews
*	Did label encoding on the target field (Sentiment)

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables. 

![alt text](https://github.com/gjariwala9/Playstore-App-Predictor/blob/master/README_IMG/rating.png "Rating")
![alt text](https://github.com/gjariwala9/Playstore-App-Predictor/blob/master/README_IMG/category.png "Categories")
![alt text](https://github.com/gjariwala9/Playstore-App-Predictor/blob/master/README_IMG/top_cat.png "Top Categories")
![alt text](https://github.com/gjariwala9/Playstore-App-Predictor/blob/master/README_IMG/word_cloud.png "Word Cloud")

## Model Building 

**For Rating Prediction**

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried six different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.   

I tried six different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression**
*	**Ridge Regression**
*	**Random Forest Regressor** 
*	**Xgboost** 
*	**ANN** 

**For Review Sentiment Classification**

First, I preprocessed the app reviews. I also split the data into train and tests sets with a test size of 20%.   

I tried four different models and evaluated them using Accuracy.  

I tried four different models:
*	**Naive Bayes**
*	**Random Forest Classifier** 
*	**Xgboost** 
*	**Logistic Regression** 

## Model performance

**For Rating Prediction**

*	**Multiple Linear Regression:** MAE = 0.34
*	**Lasso Regression:** MAE = 0.34
*	**Ridge Regression:** MAE = 0.34
*	**Random Forest Regressor:** MAE = 0.31
*	**Xgboost:** MAE = 0.33
*	**ANN:** MAE = 0.35

**For Review Sentiment Classification**

*	**Naive Bayes:** Accuracy = 51.19% 
*	**Random Forest Classifier:** Accuracy = 90.84% 
*	**Xgboost:** Accuracy = 89.45% 
*	**Logistic Regression:** Accuracy = 90.25% 

## Productionization 
In this step, I built a flask API endpoint that is hosted on a heroku webserver. The API endpoint takes list of values from user and returns an estimated Rating. Also, on providind review it will return the sentiment of it. 

**Application Link:** https://playstore-app-prediction.herokuapp.com/

![alt text](https://github.com/gjariwala9/Playstore-App-Predictor/blob/master/README_IMG/rating_form.png "Rating Form")
![alt text](https://github.com/gjariwala9/Playstore-App-Predictor/blob/master/README_IMG/rating_prediction.png "Rating Prediction")
![alt text](https://github.com/gjariwala9/Playstore-App-Predictor/blob/master/README_IMG/review_form.png "Review Form")
![alt text](https://github.com/gjariwala9/Playstore-App-Predictor/blob/master/README_IMG/review_prediction.png "Review Prediction")

