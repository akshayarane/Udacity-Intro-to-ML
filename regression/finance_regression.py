#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("/home/akshaya/udacity-ml-course-copy/udacity-ml-course/ud120-projects/tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("/home/akshaya/udacity-ml-course-copy/udacity-ml-course/ud120-projects/final_project/final_project_dataset_modified.pkl", "r") )

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
# Make predictions using the testing set
pred = reg.predict(feature_test)



score=r2_score(target_test, pred)


# The coefficients
print('slope: ', reg.coef_)
print('intercept:', reg.intercept_)
print('score: %.2f', score)

### Extract the slope (stored in the reg.coef_ attribute) and the intercept. 
### What are the slope and intercept?
### slope=5.44814029
### intercept=-102360.54329387983

### Imagine you were a less savvy machine learner, and didn’t know to test on a holdout test set. Instead, you tested on the same data that you used to train, by comparing the regression predictions to the target values (i.e. bonuses) in the training data. What score do you find? 

### score=r2_score(target_train,pred)
### score=0.0455

### Now compute the score for your regression on the test data, like you know you should. 
### What’s that score on the testing data? If you made the mistake of only assessing 
### on the training data, would you overestimateor underestimate the performance of your regression?
### score=r2_score(target_test,pred)
### score=-1.485

### Perform the regression of bonus against long term incentive--what’s the score on the test data?
### features_list = ["bonus", "long_term_incentive"]
### score=r2_score(target_test,pred)
### score=-0.593


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
#%matplotlib inline

for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
reg.fit(feature_test, target_test)
pred = reg.predict(feature_train)
print('slope: ', reg.coef_)

plt.plot(feature_train, reg.predict(feature_train), color="b") 
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
