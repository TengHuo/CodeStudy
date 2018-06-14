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

# 1,095,040REDMOND


import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

poi_count = 0
nan_count = 0
for name in enron_data:
    if enron_data[name]["poi"]:
        poi_count += 1
        if "NaN" == enron_data[name]["total_payments"]:
            nan_count += 1

print nan_count
print poi_count

# {'salary': 'NaN', 'to_messages': 'NaN', 'deferral_payments': 564348, 'total_payments': 564348, 'exercised_stock_options': 886231, 'bonus': 'NaN', 'restricted_stock': 208809, 'shared_receipt_with_poi': 'NaN', 'restricted_stock_deferred': 'NaN', 'total_stock_value': 1095040, 'expenses': 'NaN', 'loan_advances': 'NaN', 'from_messages': 'NaN', 'other': 'NaN', 'from_this_person_to_poi': 'NaN', 'poi': False, 'director_fees': 'NaN', 'deferred_income': 'NaN', 'long_term_incentive': 'NaN', 'email_address': 'james.prentice@enron.com', 'from_poi_to_this_person': 'NaN'}
