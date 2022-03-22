#!/usr/bin/env python
# coding: utf-8

# In[10]:


##### ***********FINAL PROJECT****************######


# In[ ]:


# PROJECT TITLE                   : Predict covid test result.

# MODEL                           : Logistic Regression/Binary Classification Model.

# PROJECT SHORT DESCRIPTION       : Build and train a model that can predict covid test results 
#                                   using Logistic Regression/Binary Classification.

# PROGRAMMING LANGUAGE            : Python

# EDITOR                          : Jupyter Notebook

# Exploratory data analysis (EDA) : Python scripts for EDA-marginal and GGPLOT for EDA-pairs
                                    ## GGPLOT in PYTHON ### Visualizing Data With plotnine ###
                                    ## plotnine : Python packages that provide a grammar of graphics

# GOODNESS OF FIT                 : Confusion Matrix:


# In[1]:


import pandas as pd                    ###  pandas -open source Python Library for data analysis and manipulation.

import numpy as np                     ###  numpy - Numerical Python- Python library used for working with arrays. 

from matplotlib import pyplot as plt   #### matplotlib- graph plotting library in python that used as a visualization utility


covid_analysis = pd.read_csv(r"D:\covid_analysis.csv")     #####  load CSV data file into a Pandas dataframe:
covid_analysis.head()


# In[ ]:



##  SUMMARY OF THE DATA:
    
### PersonID - ID Number of the patient/Individual.
### Age - Age (years)
### Ethnicity - Social/cultural/national group
### Ethnicity_group - Group sorted alphabetically and numbered 

                # Ethnicity             Ethnicity_Group
                # African American              	1
                # Asian                         	2
                # Hispanic                      	3
                # Latino                        	4
                # Native American/Alaskan Native	5
                # Other                         	6
                # Pacific Islander              	7
                # white                         	8

### Fever - Temperature recorded in degrees Fahrenheit 
### Asthma - Breathing issue recorded - Positive or negative
### Asthma_group - Breathing issue recorded - Positive = 1 or negative = 0
### BloodPressure - Diastolic blood pressure (mm Hg)
### HeartRate  - Beats per minute
### Glucose - Plasma glucose concentration
### BMI - Body mass index (\frac{weight}{height^2}height2weight in kg/m)

### Outcome(COVID Test Result - Class variable (0 - healthy or 1 - diabetic)


# In[42]:


covid_analysis


# In[2]:


## CHECK THE DATA TYPE OF EACH DATA

df = pd.DataFrame(covid_analysis)

print (df.dtypes)


# In[13]:


###  This data consists of PERSON/INDIVIDUAL RECORDS who have been tested for covid. 

###  The last Column in the dataset - (COVID-Test Result)
##   contains the value 0 for patients who tested negative for covid, and 1 for patients who tested positive. 
##   This is the LABEL that we will train our model to predict;

##  other columns (Age, Fever, Asthma , BloodPressure, HeartRate, Glucose, BMI and so on) are the FEATURES.

###  Seperate the FEATURES(X) FROM THE LABEL(Y).   


# In[4]:


# Seperate the FEATURES(X) FROM THE LABEL(Y). 

features = ['Age-years','Fever-temperature','Ethnicity','Ethnicity_Grp','Asthma','Asthma_Grp','BloodPressure-mm_hg','HeartRate-beatspermin','Glucose-SugarLevel','BMI-BodyMassIndex']

label = 'COVID-TestResult'

x = covid_analysis[features].values

y = covid_analysis[label].values

### Data analysis for each data values

print("             'Age','Fever','Eth','Eth_Grp',Asthma','As_Grp','BP','Heart','Sugar','BMI',  'Label'")

for n in range(0,5):
    print("Person", str(n+1), "\n  Features:",list(x[n]), "  Label:", y[n])
    
for n in range(13995,14000):
    print("Person", str(n+1), "\n  Features:",list(x[n]), "  Label:", y[n])
    


# In[ ]:


######  EXPLORATORY DATA ANALYSIS ####this is DATA Investigation...####


# In[ ]:


##Exploratory data analysis (EDA) FOR MARGINALS - USING PYTHON SCRIPTS      	df = pd.DataFrame(d)


# In[5]:


df = pd.DataFrame(covid_analysis)       # Lets call the dataframe as df 

df.loc[:,"Age-years"].mean()     ###      MEAN : sum of the values and dividing with the number of values


# In[6]:


df.loc[:,"Age-years"].median()   ### #  MEDIAN : The middle most value in a data series is called the median


# In[76]:


df.loc[:,"Age-years"].mode()     ###    MODE is the value that has highest number of occurrences in a set of data.


# In[7]:


df.loc[:,"Age-years"].var()     #### VARIANCE describe the variability of the observations from its arithmetic mean


# In[8]:


df.loc[:,"Age-years"].std()    ### STANDARD DEVIATOION measure of dispersion of observation within dataset relative 
                               ### to their mean.
                               ### It is square root of the variance 


# In[ ]:


##Exploratory data analysis (EDA) FOR PAIRS - USING GGPLOT AS BELOW


# In[ ]:


## GGPLOT in PYTHON ### Visualizing Data With plotnine ### 

## GGPLOT = Grammar of Graphics is a high-level tool
##          that allows you to create data plots in an efficient and consistent way.

## plotnine : Python packages that provide a grammar of graphics


# In[11]:


##### GGPLOT  ######

#####  Data Investigation - Individuals with Fever and covid results

from plotnine  import ggplot, aes, labs, geom_point     ## import ggplot() class,aes() and geom_line() from plotnine. 
(  
    ggplot(covid_analysis)                              ## passing the covid_analysis DataFrame to the constructor.
    + aes(x="Fever-temperature", y="COVID-TestResult")  ## AES # Set variable for each axis 
    + labs(
        x="Fever-temperature",
        y="COVID-TestResult",
        title="COVID-TestResult Versus Body-Temperature",
    )
    + geom_point()                                      ## Geometric object to use for drawing - here specified point graph
)


# In[13]:


#####  Data Investigation - Individuals with Fever and covid results also included the heartRate

from plotnine import ggplot, aes, labs, geom_point

(
    ggplot(covid_analysis)
    + aes(x="Fever-temperature", y="COVID-TestResult", color="HeartRate-beatspermin")
    + labs(
        x="Fever-temperature",
        y="COVID-TestResult",
        color="HeartRate-beatspermin",
        title=" COVID-TestResult Versus Body-Temperature - (Heart Rate Included)",
    )
    + geom_point()
)


# In[17]:


#####  Data Investigation - Individuals with Fever and covid results also included the Glucose Level

from plotnine import ggplot,aes,labs,geom_point

(
    ggplot(covid_analysis)
    + aes(x="Fever-temperature", y="HeartRate-beatspermin", color="Glucose-SugarLevel")
    + labs(
        x="Fever-temperature",
        y="HeartRate-beatspermin",
        color="Glucose-SugarLevel",
        title=" COVID-TestResult Versus HeartRate-beatspermin - (Glucose Level Included)",
    )
    + geom_point()
)


# In[18]:


from plotnine.data import mpg
from plotnine import ggplot, aes, facet_grid, labs, geom_point, theme_dark

(
    ggplot(covid_analysis)
    + facet_grid(facets="Asthma~Ethnicity")      ####  facet_grid() displays the partitions in a grid
    + aes(x="Age-years", y="HeartRate-beatspermin")
    + labs(
        x="Age-years",
        y="HeartRate-beatspermin",
        title="Result ",
    )
    + geom_point(size = -0.2)
    + theme_dark()  
)


# In[20]:


## BOX PLOT - DATA ANLYSIS 

## displays the five-number summary of a set of data

##  MINIMUM first quartile MEDIAN first quartile MAXIMUM


from matplotlib import pyplot as plt             #### matplotlib is a package in python-> import pyplot from this package

get_ipython().run_line_magic('matplotlib', 'inline')

features = ['BloodPressure-mm_hg','Glucose-SugarLevel','BMI-BodyMassIndex']
            #,'Ethnicity','HeartRate-beatsperminute','Fever-temperature']
for col in features:
    covid_analysis.boxplot(column=col, by='COVID-TestResult', figsize=(6,6))
    plt.title(col)
   #  plt.set_ylabel("Label")
plt.show()


# In[65]:


### TESTING AND TRAINING #### Split the data


# In[21]:


features = ['Age-years','Fever-temperature','Ethnicity_Grp','Asthma_Grp','BloodPressure-mm_hg','HeartRate-beatspermin','Glucose-SugarLevel','BMI-BodyMassIndex']

label = 'COVID-TestResult'

x, y = covid_analysis[features].values, covid_analysis[label].values


# In[22]:


###  scikit-learn package has train_test_split function 

### Using this package/function we can get a statistically random split of TRAINING and TEST data. 

###  split the data into 70% for TRAINING and hold back 30% for TESTING.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=0)

print ('Training cases: %d\nTest cases: %d' % (x_train.shape[0], x_test.shape[0]))

### random_state=0 is for initializing internal random number generator, which decides splitting of data into train and test

###  from 14000 cases :    Training cases - 70% : 9800      Test cases-30% : 4200


# In[67]:


###  MODEL SELECTED/CHOOSEN FOR THIS PROJECT    :  Binary Classification Model

###  ALGORITH SELECTED/CHOOSEN FOR THIS PROJECT : logistic Regression


###  NEXT STEP = Train our model by fitting the training features (X_train) to the training labels (y_train).

####  WE WILL USE Logistic Regression algorithms to train this model. 

### In addition to the training features and labels, we'll need to set a regularization parameter.

### regularization parameter - This is used to counteract any bias in the sample, and help the model
### generalize well by avoiding overfitting the model to the training data.  
    
    


# In[25]:


# Train the model

from sklearn.linear_model import LogisticRegression   ### Step 1: Import LogisticRegression from sklearn.linear_mode

# Set regularization rate

reg = 0.01

# train a logistic regression model on the training set x_train, y_train

model = LogisticRegression(C=1/reg, solver="liblinear").fit(x_train, y_train)

print (model)

## solbver liblinear — Library for Large Linear Classification. 
## This Uses a coordinate descent algorithm. Coordinate descent is based on minimizing a multivariate function by solving univariate optimization problems in a loop. In other words, it moves toward the minimum in one direction at a time.


## C is known as a "hyperparameter." -  instruct the model on how to choose parameters.
## Regularization will penalize the extreme parameters, the extreme values in the training data leads to overfitting.


# In[90]:


### Now we've trained the model using the training data, 

### Next Step test the data with the test dataset 30%  

#### we can use scikit-learn to predict.  Use the model to predict labels for our TEST set, 

###  and compare the predicted labels to the known OR ACTUAL labels:

### CHECKING TO SEE IF THE PREDICTION AND THE ACTUAL


# In[26]:


predictions = model.predict(x_test)

print('Predicted labels: ', predictions,'\n')

print('Actual labels:    ' ,y_test,'\n')


# In[92]:


### scikit-learn some metrics that we can use to evaluate the model.

### check the accuracy of the predictions -what PERCENTAGE of the labels did the model predict correctly?

### Calculate the accuracy - Mathematically it represents the ratio of the sum of true positives and true negatives out of all the predictions.


# In[27]:


from sklearn.metrics import accuracy_score        

print('Accuracy: ', accuracy_score(y_test, predictions))


# In[94]:


### The accuracy is returned as a decimal value - a value of 1.0 MEANS the model got 100% of the predictions CORRECT!!!! 

### while an accuracy of 0.0 is, well, is not good at all !!!!!


# In[95]:


##  SUMMARY

####   prepared the data by splitting it into TEST and TRAIN datasets, and applied logistic regression 

####   Our model was able to predict whether an Individual(Person) had COVID with good accuracy.

####   Accurancy = 0.99 ---This is Good!!!

###    But What are the other ways that we can check if this model is good enough to use!!!


# In[16]:


from sklearn. metrics import classification_report

print(classification_report(y_test, predictions))


# In[ ]:


## Classification report is used to measure the quality of predictions from a classification algorithm.

## How many predictions are True and how many are False. 
## More specifically, True Positives, False Positives, True negatives 
## and False Negatives are used to predict the metrics of a classification report. 


# In[97]:


###   Precision: Of the predictions the model made for this class, what proportion were correct?

      ## Precision is the ability of a classifier not to label an instance positive that is actually negative.

###   Recall: Out of all of the instances of this class in the test dataset, how many did the model identify?

      ### Recall is the ability of a classifier to find all positive instances.

###   F1-Score: An average metric that takes both precision and recall into account. 

      ### The F1 score is a weighted mean of precision and recall such that the best score is 1.0 and the worst is 0.0. 

##    Support: How many instances of this class are there in the test dataset?

### The classification report includes averages for these metrics,weighted average allows the imbalance in the number of cases of each class.

### Of all the person the model predicted are COVID, how many are actually COVID?

### Of all the patients that are actually COVID, how many did the model identify AS COVID?


# In[17]:


from sklearn.metrics import precision_score, recall_score

print("Overall Precision:",precision_score(y_test, predictions),'\n')


print("Overall Recall:",recall_score(y_test, predictions))


# In[99]:


## The precision and recall metrics are derived from four possible prediction outcomes:

### True Positives: The predicted label and the actual label are both 1.
### False Positives: The predicted label is 1, but the actual label is 0.
### False Negatives: The predicted label is 0, but the actual label is 1.
### True Negatives: The predicted label and the actual label are both 0.


# In[100]:


## sklearn.metrics.confusion_matrix function to find these values for a trained classifier:


# In[28]:


from sklearn.metrics import confusion_matrix

# Print the confusion matrix

cm = confusion_matrix(y_test, predictions)
print (cm)

## matrix between y test data versus the predictions

##   True Positives     False Negatives
##   False Positives    True Negatives

##   Example : True Positives: Out of 3234 actual positive cases, in 3228 cases the model predicted positive.


# In[102]:


##  Statistical machine learning algorithms like logistic regression is based on probability; 

## binary classifier prediction is probability that the label is true (P(y)) and the probability that the label is false (1 - P(y)).

### A threshold value of 0.5 is used to decide whether the predicted label is a 1 (P(y) > 0.5) or a 0 (P(y) <= 0.5). 

### if gretaer than 0.5  = Predicted lable = 1

### if less than 0.5  = Predicted lable = 0


# In[32]:


import joblib

# Save the model as a file

filename = './covid_model.pkl'

joblib.dump(model, filename)


# In[119]:


###  When we have some new observations for which the label is unknown,

### we can load the model and use it to predict values for the unknown label:


# In[43]:


## TESTING SAMPLE 1

# Load the model from the file

model = joblib.load(filename)

# predict on a new sample

# The model accepts an array of feature arrays (so you can predict the classes of multiple Individuals/persons
## in a single call). # Creating an array with a single array of features, representing one Person

print ('Enter the New sample data that you want to check: {}'.format(list(x_new[0])),'\n')

x_new = np.array([[30, 96, 1, 1, 60, 97, 79, 45.33189957]])

# Get a prediction   # Get a prediction Prob as well

pred = model.predict(x_new)

pred_prob = model.predict_proba(x_new)

# The model returns an array of predictions - one for each set of features submitted
# In our case, we only submitted one patient, so our prediction is the first one in the resulting array.

print('Predicted class is {}'.format(pred[0]),'\n')
print('Predicted - Covid Testing Result = 0 = Negative','\n' )


# In[44]:


## TESTING SAMPLE 2


# Load the model from the file

model = joblib.load(filename)

# predict on a new sample

# The model accepts an array of feature arrays (so you can predict the classes of multiple Individuals/persons
## in a single call). # Creating an array with a single array of features, representing one Person

print ('Enter the New sample data that you want to check: {}'.format(list(x_new[0])),'\n')

x_new = np.array([[40, 101, 3, 1, 69, 132, 136, 40.81699943]])

# Get a prediction   # Get a prediction Prob as well

pred = model.predict(x_new)

pred_prob = model.predict_proba(x_new)

# The model returns an array of predictions - one for each set of features submitted
# In our case, we only submitted one patient, so our prediction is the first one in the resulting array.

print('Predicted class is {}'.format(pred[0]),'\n')

print('Predicted - Covid Testing Result = 1 = Positive','\n' )


# In[1]:


####  THIS COMPLETES THE PROJECT PRESENTAION AND DEMO SESSION ##### 


# In[ ]:


####  thanks for listening to this presentaion ##### 

