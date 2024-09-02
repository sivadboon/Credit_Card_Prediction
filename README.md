# Credit Card Prediction Repository
<hr style="border:2px solid gray"> </hr>

![alt text](https://github.com/enlihhhhh/credit_card_prediction/blob/main/sc1015%20Cover%20Template.JPG)

## Contributors
- @enlihhhhh Data Cleaning, Data Extraction, XGBoost Classification
- @tomokiteng Data Visualisation, Decision Tree
- @sivadboon Logistic Regression, Random Forest
<hr style="border:2px solid gray"> </hr>

## About
### Synopsis
We are a group of NTU students undertaking the task of predicting the Credit Card Approval for our SC1015 module Mini-Project (Introduction to Data Science and Artificial Intelligence). We have taken our dataset from kaggle, under the **'Credit Card Approval Prediction'** [here](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction).
### Problem Definition
How does the various data submitted by Credit Card Applicants determine their credit scores (i.e. Clients’ Risk) based on prediction of future defaults and credit card loans?
### Defining Risks
Our team's objective is to distinguish the good clients from the bad clients based on the time taken for them to repay their loans. The risk is based on the time taken for them to repay their loans. We have defined the good clients and bad clients as follows:
> Good Clients
> - Impose a low risk on banks
> - Either no loans or able to repay their loans within 29 days

> Bad Clients
> - Impose a high risk on banks
> - Takes longer than 30 days to repay their loans or default on their loans
### Rationale
Our team's reason for choosing this particular dataset is as follows:
> Context
> - Usage of Credit card is extremely prevelant in today's day and age
> - However, given that credit is built on the system of trust, it is important for the banks to known which are the clients that impose a higher risk so that they can be highlighted

> Dataset
> - The dataset for the client's credit card approval is extremely exhaustive and delineate
> - Especially so as financial firms would collect as many information as possible to determine if the client impose a high risk
> - After all, the credit system is akin to a short term loan

> Prevalence
> - In 2019, credit card stats showed that the average consumer credit card debt was higher than it ever had been
> - 2.8 billion credit cards in use worldwide
> - 70% of people have at least one credit card

<hr style="border:2px solid gray"> </hr>

## Content page
### Repository Content
1. [Main Project File](https://github.com/enlihhhhh/credit_card_prediction/blob/main/credit_card_prediction_MiniProject.ipynb) 
   - Problem Definition (Introduction and Content Explanation)
   - Data Wranggling and Data Cleaning (Dealing with duplicates, data types and null values etc.)
   - Exploratory Data Analysis (Exploring Numerical and Categorical variables, as well as further data cleaning and aggregation)
   - Data merging
   - Data Visualisation (For all Response variables for a full picture)
   - Machine Learning (SMOTE, Decision Tree, Random Forest, Logistic Regression, XGBoost Classification)
   - Conclusion 
2. [Project Slides Brief](https://github.com/enlihhhhh/credit_card_prediction/blob/main/Credit%20Card%20Prediction%20Slides.pptx)
   - Introduction to Dataset (Background information and Problem Definition)
   - Data Engineering (Data wrangling, EDA, and Data insights)
   - Core Analysis (Machine Learning models and new techniques learnt)
   - Conclusion and Outcome (Data driven insights and recommendations)
3. [Project Slides Transcript](https://github.com/enlihhhhh/credit_card_prediction/blob/main/Credit%20Card%20Prediction%20SC1015%20%E2%80%93%20Presentation%20Transcript.pdf)
   - Introduction to Dataset (Background information and Problem Definition)
   - Data Engineering (Data wrangling, EDA, and Data insights)
   - Core Analysis (Machine Learning models and new techniques learnt)
   - Conclusion and Outcome (Data driven insights and recommendations)
4. [Kaggle Data source](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
   - Introduction to Dataset (Background information and Problem Definition)
   - Data Engineering (Data wrangling, EDA, and Data insights)
   - Core Analysis (Machine Learning models and new techniques learnt)
   - Conclusion and Outcome (Data driven insights and recommendations)

### Models used
1. Decision Tree
2. Random Forest
3. Logistic Regression
4. XGBoost Classification
<hr style="border:2px solid gray"> </hr>

## Lesson Learnt
### Through the project
- SMOTE (Imbalanced dataset handling)
- Onehot encoding (Handling of Categorical Data)
- Random Forest from sklearn (Machine Learning model)
- Logistic Regression from sklearn (Machine Learning model)
- XGBClassifier from XGBoost (Machine Learning model)

### Data Driven Insights and Interesting Concepts
1. SMOTE
    - Given that our data was extremely imbalanced with most of our clients being classified as a 'good client', most models did not fit well initially. 
    - To solve this issue, we used the concept of SMOTE (Synthetic Minority Oversampling Technique). 
    - This allows additional points to be synthetically created which allows models to more effectively be trained to identify a 'bad client'
 
 2. K-fold cross validation
    - It can be used to select the best parameters such as number of trees, depth of the tree, and the rate at which the gradient boosting learns. 

3. Limits of Linear Regression -> Logistic Regression
    - In our course, we learnt that we can use linear regression to predict numerical and continuous response using the independent variables. 
    - However, the same cannot be said for categorical response. Hence, after some research online, we found out that we can use logistic regression to predict categorical response. 
    
4. Limits of Decision Tree -> Random Forest
    - While decision trees are one of the most used models in predicting categorical variables, they may not return the model with the highest prediction accuracy. 
    - Hence, we use Random forest to ensure the prediction to be more accurate (i.e. higher prediction accuracy) 
    - This is done through an esemble learning method, which is operated by constructing a multitude of decision trees and return the mean prediction of the individual trees. 
    
5. Limits of Decision Tree -> XGBoost
    - While decision trees are one of the most used models in predicting categorical variables, they often exhibit highly variable behaviour which may result in errors. 
    - Hence, we use boosting to ensure that the trees are built sequentially such that each subsequent tree will reduce the errors of the previous tree. 
    - In contrast to the Random Forest technique that we implemented earlier in which trees are grown to their maximum extent, XGBoost make uses of trees with fewer splits. Such small trees are easily comprehended and readble. 
    
6. Outcome
    - After the various steps we have taken to answer our problem definition, we have found that the machine learning model best suit our dataset was XGBoost with a model accuracy of 83.3%
    - XGBoost would be very useful model for the banks to predict if a client were a good client or bad client; given the risks of client defaulting on their payments upon credit card which banks should consider prior to the approval of their credit cards.

### Areas for improvements
### Extra Improvements : Using XGBoost (rfe.ranking) to determine the Best Predictors for our Response: 
For our extra improvements, we decided to use XGBoost to determine what are the best predictors for our response.

**The code we used for our Extra Improvements are as follows:**
```
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

target_df = X_train_bal.copy()
cols = list(target_df.columns)
XGB_model = XGBClassifier() 
rfe = RFE(XGB_model)

X_rfe = rfe.fit_transform(X_train_bal,y_train_bal.GOOD_OR_BAD_CLIENT.ravel())

XGB_model.fit(X_rfe,y_train_bal)
temp_df = pd.Series(rfe.support_,index=cols)
selected_features = temp_df[temp_df==True].index
print(rfe.ranking_) # gives the ranking of all the variables, 1 being the most important
print(selected_features) # prints out the columns which are the most important
```

we can see that the following Predictor variables have the highest importance affecting the accuracy of the model:
- CNT_CHILDREN
- AMT_INCOME_TOTAL
- AGE_YEARS
- YEARS_EMPLOYED
- NAME_FAMILY_STATUS

The rest of the variables are not a good estimate even though they are in the list as not all types were included in the list

### Further Improvements
We used XGBoost (rfe.ranking) to determine what is the following set of Predictor variables that had the highest importance in affecting the accuracy of the model; id est key factors that determined if our client was a 'good client' or a 'bad client'.

However, we can further improve on our Predictor variables through the use of optimal feature selection, which would allow us to better get an answer for the best set of variables for models; id est better key factors in determining if our client was a 'good client' or a 'bad client'.
<hr style="border:2px solid gray"> </hr>

### References
- Dataset References https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
- Slides Skeletal template https://slidesgo.com/
- Credit card information https://shiftprocessing.com/credit-card/
- Consumer debt information https://www.forbes.com/sites/elenabotella/2020/02/13/credit-card-debt-all-time-high-inflation-recession/?sh=42e6b569c323
- Credit card statistics https://rcbcbankard.com/blogs/credit-card-statistics-global-facts-data-and-figures-16
