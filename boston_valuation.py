# Make an import of all resources needed.

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data= boston_dataset.data, columns = boston_dataset.feature_names)
# data.head()

features = data.drop(['INDUS', 'AGE',], axis = 1)
# features.head()
# this s two dimantional  dataset

# creating log prces for our dataset
# log prices are one dimension al and has to be madde into two
log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])

CRIME_IDX = 0
ZN_INX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

#property_stats = np.ndarray(shape=(1,11))
#property_stats[0][CRIME_IDX] = features['CRIM'].mean()
#property_stats[0][ZN_INX]  = features['ZN'].mean()
#property_stats[0][CHAS_IDX] =  features['CHAS'].mean()

property_stats = features.mean().values.reshape(1,11)
property_stats


regr = LinearRegression().fit(features,target)
fitted_vals = regr.predict(features)
# Calculate the MSE  nad RMSE using sklearn

MSE = mean_squared_error(target,fitted_vals)
RMSE = np.sqrt(MSE)

 # RMSE

 # function that calculates log prices

def get_log_estimate(nr_room,
                    students_per_classroom,
                    next_to_river = False,
                    high_confidence = True):
    #Configure property
        property_stats[0][RM_IDX] = nr_room
        property_stats[0][PTRATIO_IDX] = students_per_classroom

    #Configure property
        if next_to_river:
            property_stats[0][CHAS_IDX] = 1
        else :
            property_stats[0][CHAS_IDX] = 0

        log_estimamte = regr.predict(property_stats)[0][0]

        if high_confidence:
            upper_bond = log_estimamte + 2* RMSE
            lower_bond = log_estimamte - 2* RMSE
            interval = 95

        else:
            upper_bond = log_estimamte +  RMSE
            lower_bond = log_estimamte -  RMSE
            interval = 68

        return log_estimamte,upper_bond,lower_bond, interval

get_log_estimate(3,20, next_to_river = True, high_confidence = False)


ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)

log_set,upper,lower,conf = get_log_estimate(9,students_per_classroom=15,
                                            next_to_river = False,
                                            high_confidence = False)

#Convert to todays dollar
dollar_est = np.e**log_set * 1000 * SCALE_FACTOR
dollar_hi = np.e**lower * 1000 * SCALE_FACTOR
dollar_low = np.e**upper * 1000 * SCALE_FACTOR

# Round the dollar values to nearest thousand
round_est = np.around(dollar_est,-3)
round_hi = np.around(dollar_hi,-3)
round_low = np.around(dollar_low,-3)


print(f'The estimated property value is {round_est}')
print(f'At {conf} % confidence the valuation range is')
print(f'USD {round_hi} at the lower end to USD  {round_low } at the high end.')

def get_dollar_value(rm,ptratio,chas=False,large_range=True):

    """
    Estimate the price of a property in boston.
    Keywords argumenets:

    rm -- n umber of rooms in the property
    ptratio -- number of students per teacher in the classroom for the schoolin the area
    chas -- True if the property in next to chase rive, False Otherwise
    large_range --  True for a 95 % prediction interval, False for a 68% interval

    """
    if rm < 1 or ptratio < 1:
            print(f'Thats is unrealistic Try Again.')
            return

    log_set,upper,lower,conf = get_log_estimate(rm,students_per_classroom = ptratio ,
                                                next_to_river = chas,
                                                high_confidence = large_range)

    #Convert to todays dollar
    dollar_est = np.e**log_set * 1000 * SCALE_FACTOR
    dollar_hi = np.e**lower * 1000 * SCALE_FACTOR
    dollar_low = np.e**upper * 1000 * SCALE_FACTOR

    # Round the dollar values to nearest thousand
    round_est = np.around(dollar_est,-3)
    round_hi = np.around(dollar_hi,-3)
    round_low = np.around(dollar_low,-3)


    print(f'The estimated property value is {round_est}')
    print(f'At {conf} % confidence the valuation range is')
    print(f'USD {round_hi} at the lower end to USD  {round_low } at the high end.')
