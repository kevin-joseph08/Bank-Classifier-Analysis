# Portugese Bank : Direct Marketing Campaign 
### Determine parameters that can help future campaign to attract client to subscribe to a term deposit 
**Assignment:** Compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines) and share observations & recommendations

Dataset is from UCI Dataset: Link: https://archive.ics.uci.edu/dataset/222/bank+marketing

<img width="948" alt="image" src="https://github.com/user-attachments/assets/56564531-c135-47e2-a418-2f333bc465bf">

## Overview:
This is a practical application assignment as part of the UC Berkeley Haas AI-ML Course. The goal is to compare the performance of the classifiers (k-nearest neighbors, logistic regression, decision trees, and support vector machines). The dataset is related to the marketing of bank products over the telephone and avaialble on the UCI Website

## Business Understanding:

Comparing_Classifiers_Portugese_Bank/CRISP-DM-BANK.pdf
According to the article (link above), there were **17 campaigns** between **May 2008 and Nov 2010**. These phone campaings focused on offered **long-term deposits** with good interest rates. The **success** was determined if the **customer subscribed** to the long-term deposits.

**The focus for the bank is to identify key attributes that can help improve their success rate and attract customers to subscribe to long-term deposits.**

## Understanding the Data

The dataset comes with 21 attributes. There are no null values so all rows have information that can be utilized for data analysis and model evaluation.
There are 41188 records for us to analyze. Few of the attributes are numeric while the others are categorical.

The dataset is broken into 4 sections:

#### Bank Client data
- Age
- Job : type of job
- marital : marital status
- education
- default: has credit in default?
- housing: has housing loan?
- loan: has personal loan?

#### Information related with the last contact of the current campaign
- contact: contact communication type
- month: last contact month of year
- day_of_week: last contact day of the week
- duration: last contact duration, in seconds

#### Other Attributes
- campaign: number of contacts performed during this campaign and for this client
- pdays: number of days that passed by after the client was last contacted from a previous campaign
- previous: number of contacts performed before this campaign and for this client
- poutcome: outcome of the previous marketing campaign

#### Social and Economic context attributes
- emp.var.rate: employment variation rate - quarterly indicator (numeric)
- cons.price.idx: consumer price index - monthly indicator (numeric)
- cons.conf.idx: consumer confidence index - monthly indicator (numeric)
- euribor3m: euribor 3 month rate - daily indicator (numeric)
- nr.employed: number of employees - quarterly indicator

#### Target Variable
- y - has the client subscribed a term deposit? (yes or no)

#### Missing Data needing special attention
- There are a few records that has a value of 'unknown' and it represents missing data. The attributes with these missing values are: `job`, `education`, `default`, `housing`, and `loan`.
- The attribute `pdays` has a value of 999 indicating that the client was not previously contacted
- The attribute `duration` highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

## Exploratory Data Analysis

There are a total of 13 attributes that have less than or equal to 12 unique values. These attributes are:
```Column          Count
contact         2
default         3
housing         3
loan            3
poutcome        3
marital         4
day_of_week     5
education       8
previous        8
month           10
emp_var_rate    10
nr_employed     11
job             12
```
#### Day of the Week
Looking at the `day_of_week`, we find that it is equally distributed. There is no uniqueness that differentiates any specific day of the week. We can derive that the data will not have any weightage on the model prediction. This column will be removed from the dataframe and not be part of further analysis

<img width="748" alt="image" src="https://github.com/user-attachments/assets/86f2a8f4-4dce-4230-a7c2-349a9522f61b">

#### Contact
The contact values are `cellular` and `telephone`. We find that 64% of the customers have `cellular` compared to 36% with regular telephone (landline) service. Further analysis shows that this celular data has a correlation to customers accepting the marketing promotion.

<img width="593" alt="image" src="https://github.com/user-attachments/assets/772c9b8a-fed2-4c05-b5fc-08385f2bd81a">

#### Default
One of the attribute is to checks if the customer has their credit in default. 
<img width="596" alt="image" src="https://github.com/user-attachments/assets/226c6f5a-8219-4f83-9469-e5f2562d5ffa">


<img width="703" alt="image" src="https://github.com/user-attachments/assets/8ac42fb2-70e0-46b6-bf8b-193309ef5f8b">

As you can see from the above charts, the atrribute `default` has some correlation to customers accepting the marketing promotion.

#### Pair Plot of all Categorical attributes with Target Variable

Here's a view of the pair plot when compared to the target variables

![Pair Plot Portugese Bank](https://github.com/user-attachments/assets/a54c925a-dc38-412e-a2e3-8b90ea1e0733)

## Feature Engineering

Now that we have a good idea of all the attributes, let's review the categorical attributes and see if we can encode them.

#### Encoding Categorical Attributes
We have a few attributes that can be encoded. To reduce the complexity, I converted these using Label Encoding.

contact         2
default         3
housing         3
loan            3
poutcome        3
marital         4
education       8
month           10
job             12
y               2 (Target Variable)

#### Pearson's Correlation of all attributes

With the new set of numerical attributes, I ran the Pearson Correlation. Below is the result of the correlation between all the numerical variables.

![Pearsons Correlation Portugese Bank](https://github.com/user-attachments/assets/f00126e7-a9a0-4735-b30a-0844a80aa3a6)

## Model Evaluation

Using the refined dataset, I split this into 80% for training and 20% for testing.
To create standardized results for each of the model, I created a series of functions and called these functions for each of the models.

**Functions created:**
- **Print Performance:** This will print the performance results of each model. It will print the accuracy, recall, precision, f1 scores.
- **Print Confusion Matrix:** This will print the confusion matrix and the associated values of True Positive, True Negative, False Positive, and False Negative
- **Print ROC-AUC Scores:** This will plot the ROC-AUC Curve and print the ROC-AUC score.
- **Evaluate Function:** This will use either the default setting or the hyperparameter to call the model. Perform the model fit, predict, and calculate and print the processing time, performance, confusion matrix, and the ROC-AUC curve and scores.

### Model Comparison

I created a baseline of the model using `Dummy Classifer` and then evaluated the following models without any hyperparameter tuning.

The Confusion Matrix for the Dummy Classifier (as expected) is shown below.

<img width="596" alt="image" src="https://github.com/user-attachments/assets/5d942717-1f6b-4db6-b006-2688d69df0b8">

The ROC-AUC Curve for the Dummy Classifier will be a straight line in the middle as shown below.

<img width="877" alt="image" src="https://github.com/user-attachments/assets/b97b730f-7ef0-407b-8a89-a8ecdf288fce">

### Initial Model Comparison : Without Hyperparameter Tuning (using Default Settings of each model)

- **Dummy Classifier**
- **Logistic Regression**
- **Decision Tree Classifier**
- **K Nearest Neighbor Classifier**
- **Support Vector Machines**

Based on the analysis of the refined dataset, the results from these models were as folows:

#### Results from Model Evaluation using Default Settings for each Model

<img width="1390" alt="image" src="https://github.com/user-attachments/assets/2e339f36-a455-49e2-8c4f-16c9edc3cab4">

#### Confusion Matrix using Default Settings for each Model

The associated Confusion Matrix for these models (excluded Dummy Classifier) are as shown below. 

![Confusion Matrix Comparison for 4 Models](https://github.com/user-attachments/assets/3af19840-6677-4318-87d9-e79fb00bda3c)

#### ROC AUC Curve using Default Settings for each Model

The associated ROC AUC Curve for each of these models (excluding Dummy Classifier) are as shown below.

![ROC-AUC Curve Comparison for 4 Models](https://github.com/user-attachments/assets/e4d24f70-6d8d-4e29-a14d-169a217343ac)

#### Observation:
- Based on the results shown above, we can see that Logistic Regression and Support Vector Machines have a very good accuracy score of 0.91  
- However, we also see that Support Vector Machines takes 118 seconds to process 7186 records while Logistic Regression takes only 0.65 seconds for 7136 records.  
- Decision Tree Classifer and K Nearest Neighbor have a fairly lower accuracy score with Decision Tree Classifier getting fewer items correct.  
- Looking at the performance, K Nearest Neighbor has the best time while maintaining a competitive accuracy score. 

### Improved Model Comparison : With Hyperparameter Tuning

#### Feature Engineering before model comparison

During the initial run, I found that some of the features do not have a strong correlation and can be eliminated. This can improve the performance when we tune the hyperparameters.

#### Hyperparameter Selected:

- **LogisticRegression:**

```
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'saga']
```

- **Decision Tree Classifier:**

```
    'Decision Tree Classifier': {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None, 10, 20, 30, 40, 50],
        'classifier__min_samples_split': [2, 5, 10]
```
- **K Nearest Neighbor Classifier:**

```
    'K Nearest Neighbor Classifier': {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
```
- **Support Vector Machines:**

```
    'Support Vector Machines': {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
```


With the further refined dataset, below are the results for the four models:

#### Confusion Matrix using Hyperparameter Tuning for each Model

The associated Confusion Matrix for these models are as shown below.

![Optimized Confusion Matrix Comparison for 4 Models](https://github.com/user-attachments/assets/2d52b408-8f48-4de9-8968-361f73d854df)

#### Observation:
Adding hyperparameters, I was able to see a much better result for all 4 models. 

## Recommendation:
Use Logistic Regression with hyperparameters as the precision is higher, recall is higher, and ROC AUC score is also higher. The overall time it takes to process Logistic Regression is much lower than all others making it the best option among the 4 models.

## Future Questions:
A few questions we can ask about the dataset and the campaign are:
1. How many days before the campaign should the bank contact the customers. The duration of the call and the pdays and previous days in the current dataset is not providing enough information to determine the success of the campaign
2. While we see a negative correlation around employment variation rate, it does not translate to any meaningful decisions.
3. There is a good correlation between the duration of the call and contact type. Customers with cellphones show higher reach. The campaign can focus on improving the success by focusing on customers with cellphones
