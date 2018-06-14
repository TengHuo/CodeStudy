#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = []
    for index in range(len(predictions)):
        error = (predictions[index][0] - net_worths[index][0])**2
        errors.append((index, error))

    errors = sorted(errors, cmp=lambda x, y:cmp(x[1], y[1]))

    reserve_data_range = range(0, int(len(errors)*0.9))
    for index in reserve_data_range:
        # print "AGES:"
        # print ages[errors[index][0]]
        new_data = (ages[errors[index][0]], net_worths[errors[index][0]], errors[index][1])
        cleaned_data.append(new_data)

    # print ages
    # print cleaned_data
    return cleaned_data

