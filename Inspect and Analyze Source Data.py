"""
This part of code is to inspect and analyze source historical data with purposes of:
1. Understand the data, inform the team of key data variable descriptions.
2. Ensure the information provides is suitable for generating clear and meaningful insights.
3. Explore dataset to find any unusual values and outliers.
4. Consider which variables are most useful to build predictive models.
5. Considered potential interactions between the two chosen variables.
6. prepare for future exploratory data analysis.
"""

import pandas as pd       #library exercise for buidling dataframes
import numpy as np        #numpy is imported with pandas

# Load dataset into dataframe
df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')
print("done")

df.head(10)
df.info()
df.describe()

# Sort the data by trip distance from maximum to minimum value
df_sort = df.sort_values(by=['trip_distance'], ascending=False)
df_sort.head(10)

# Sort the data by total amount and print the top 20 values
#total_amount_sorted = df.sort_values(by=['total_amount'], ascending=False)['total_amount']
total_amount_sorted = df.sort_values(['total_amount'], ascending=False)['total_amount']
total_amount_sorted.head(20)

# Sort the data by total amount and print the bottom 20 values
total_amount_sorted.tail(20)

# How many of each payment type are represented in the data?
df['payment_type'].value_counts()

# What is the average tip for trips paid for with credit card?
avg_cc_tip = df[df['payment_type'] == 1]['tip_amount'].mean()
print('Avg. cc tip:', avg_cc_tip)

# What is the average tip for trips paid for with cash?
avg_cc_tip = df[df['payment_type'] == 2]['tip_amount'].mean()
print('Avg. cc tip:', avg_cc_tip)

# How many times is each vendor ID represented in the data?
df['VendorID'].value_counts()

# What is the mean total amount for each vendor?
#df.groupby(['VendorID'])[['total_amount']].mean()
df.groupby(['VendorID']).mean(numeric_only=True)[['total_amount']]   #with double [[]] prints out html format

# Filter the data for credit card payments only
#cc_payments = df.loc[df['payment_type'] == 1]
credit_card = df[df['payment_type'] == 1]

# Filter the credit-card-only data for passenger count only
#df.loc[df['payment_type'] == 1]['passenger_count'].value_counts()  #this is to count how many rows for each number of passengers
credit_card['passenger_count'].value_counts()

#df.loc[df['payment_type'] == 1, ['passenger_count']]  #this provides the 15265 rows(credit-card-paid) × 1 columns

# Calculate the average tip amount for each passenger count (credit card payments only)
#df[df['payment_type'] == 1].groupby(['passenger_count']).mean(numeric_only=True)[['tip_amount']]

credit_card.groupby(['passenger_count']).mean(numeric_only=True)[['tip_amount']]

#After looking at the dataset, the two variables that are most likely to help build a predictive model for taxi fares
#are total_amount and trip_distance because those variables show a picture of a taxi cab ride.

#The identified unusual values are trips that are a short distance but have high charges associated with them,
#as shown in the total_amount variables.
