#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np

enron_data = pickle.load(open("/home/akshaya/udacity-ml-course-copy/udacity-ml-course/final_project/final_project_dataset.pkl", "rb"))

print "Total number of people in Enron data:", len(enron_data)
#Number of people:146
print "Total number of features available for each person in Enron data:", len(enron_data["SKILLING JEFFREY K"])
#Ans=21

#Total number of POI in the data:
    people = 0
    count_poi = 0
for each in enron_data:
    people = people + 1
    if enron_data[each]['poi'] == 1:
        count_poi += 1
print count_poi      
#Ans: count_poi=18
       
    
poi_reader = open('/home/akshaya/udacity-ml-course-copy/udacity-ml-course/ud120-projects/final_project/poi_names.txt', 'r')    
poi_reader.readline()
#'http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm\n'
poi_reader.readline()
#'\n'
poi_count = 0
for poi in poi_reader:
    poi_count += 1

print poi_count
#Ans=35

#What might be a problem with having some POIs missing from our dataset?
#We might not have considered some of the features. Not enough data points. 
#What is the total value of the stock belonging to James Prentice?
enron_data["PRENTICE JAMES"]['total_stock_value']
#Ans=1095040

#How many email messages do we have from Wesley Colwell to persons of interest?
enron_data["COLWELL WESLEY"]['from_this_person_to_poi']
#Ans=11

#What’s the value of stock options exercised by Jeffrey K Skilling?
enron_data["SKILLING JEFFREY K"]['exercised_stock_options']
#Ans=19250000

#Which of these schemes was Enron not involved in?
#selling assets to shell companies at the end of each month, and buying them back at the beginning of the next month to hide accounting losses
#causing electrical grid failures in California
#illegally obtained a government report that enabled them to corner the market on frozen concentrated orange juice futures-no
#conspiring to give a Saudi prince expedited American citizenship-no
#a plan in collaboration with Blockbuster movies to stream movies over the internet

#Who was the CEO of Enron during most of the time that fraud was being perpetrated?Jeffrey Skilling
#Who was chairman of the Enron board of directors? Kenneth Lay
#Who was CFO (chief financial officer) of Enron during most of the time that fraud was going on?Andrew Fastow

#Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of “total_payments” feature)?
enron_data["SKILLING JEFFREY K"]['total_payments']
enron_data["LAY KENNETH L"]['total_payments']
enron_data["FASTOW ANDREW S"]['total_payments']

#LAY KENNETH L

#How much money did that person get?
#103559793

#For nearly every person in the dataset, not every feature has a value. How is it denoted when a feature doesn’t have a well-defined value?
print enron_data["SKILLING JEFFREY K"]
#Ans=NaN

#How many folks in this dataset have a quantified salary? What about a known email address?
print len(dict((key, value) for key, value in enron_data.items() if value["salary"] != 'NaN'))
#Ans=95

print len(dict((key, value) for key, value in enron_data.items() if value["email_address"] != 'NaN'))
#Ans=111

#How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments? 
a = len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN'))
print a
#Ans=21

21.0/146.0
#Ans=14.3%

#How many POIs in the E+F dataset have “NaN” for their total payments? What percentage of POI’s as a whole is this?
b = dict((key, value) for key, value in enron_data.items() if value["poi"] == True)
print len(b)

c = len(dict((key, value) for key, value in b.items() if value["total_payments"] == 'NaN'))
print c
#Ans=18
#0