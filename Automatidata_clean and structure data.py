"""The purpose of this project is to conduct exploratory data analysis on a provided data set.
perform further EDA on this data with the aim of learning more about the variables.

The goal is to clean data set and create a visualization.
"""

# Import packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns

# Load dataset into dataframe
df = pd.read_csv('2017_Yellow_Taxi_Trip_Data.csv')
df.head(10)
df.size
df.describe()
df.info()

# Convert data columns to datetime
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
#df.info()

# Create box plot of trip_distance
plt.figure(figsize=(8,2))
sns.boxplot(data=df, x=df['trip_distance'], fliersize=1)
plt.title('Trip Distance')

# Create histogram of trip_distance
plt.figure(figsize=(10,5))
sns.histplot(df['trip_distance'], bins=range(0, 34, 1))
plt.title('Trip Distance Histogram')

# Create box plot of total_amount
plt.figure(figsize=(7,2))
sns.boxplot(data=df, x=df['total_amount'], fliersize=1)
plt.title('Total Amount')

# Create histogram of total_amount
plt.figure(figsize=(12,6))
ax=sns.histplot(df['total_amount'], bins=range(-10,101,5))
ax.set_xticks(range(-10,101,5))
ax.set_xticklabels(range(-10,101,5))

# Create box plot of tip_amount
plt.figure(figsize=(7,2))
plt.title('tip_amount')
sns.boxplot(x=df['tip_amount'], fliersize=1)

# Create histogram of tip_amount
plt.figure(figsize=(12,6))
plt.title('tip amount histogram')
ax =sns.histplot(df['tip_amount'], bins=range(0,21,1))
ax.set_xticks(range(0,21,1))
ax.set_xticklabels(range(0,21,1))

# Create histogram of tip_amount by vendor (julie: use hue)
plt.figure(figsize=(12,6))
plt.title('Tip amount by vendor histogram')
ax = sns.histplot(data=df, x='tip_amount', bins=range(0,21,1),
                  hue='VendorID',  #if you use hue, you need to use data=df in the first argument, or error
                  multiple='stack',
                  palette='pastel')
ax.set_xticks(range(0,21,1))
ax.set_xticklabels(range(0,21,1))

# Create histogram of tip_amount by vendor for tips > $10
plt.figure(figsize=(12,6))
plt.title('Tip amount over ten by vendor histogram')
ax = sns.histplot(data=df[df['tip_amount'] > 10], x='tip_amount', bins=range(10,21,1),
                  hue='VendorID',
                  multiple='stack',
                  palette='pastel')
ax.set_xticks(range(10,21,1))
ax.set_xticklabels(range(10,21,1))

df['passenger_count'].value_counts()

# Calculate mean tips by passenger_count (Julie: by each group, calculate its own mean tips)
#mean_tips_by_passenger_count = df.groupby(['passenger_count']).mean()[['tip_amount']]  #exemplar
mean_tips_by_passenger_count = df.groupby(['passenger_count'])[['tip_amount']].mean()  #mine
mean_tips_by_passenger_count

# Create bar plot for mean tips by passenger count
data = mean_tips_by_passenger_count.tail(-1)
pal = sns.color_palette("Greens_d", len(data))
rank = data['tip_amount'].argsort().argsort() #julie: one argsort() is descending, two argsort() is as ascending
plt.figure(figsize=(12,7))
ax = sns.barplot(x=data.index, y=data['tip_amount'],
           palette=np.array(pal[::-1])[rank])
mean_val = df['tip_amount'].mean()
ax.axhline(mean_val, ls='--', color='red', label='global mean')
ax.annotate(
    f'Mean: {mean_val:.2f}',               # Text to display (formatted to 2 decimal places)
    xy=(0.98, mean_val),                  # Position: 98% across the X-axis, exact mean on Y-axis
    xycoords=ax.get_yaxis_transform(),    # Uses axis fraction for X, data coordinates for Y
    xytext=(0, 5),                        # Shifts text 5 points UP so it doesn't touch the line
    textcoords='offset points',           # Tells matplotlib to interpret xytext as pixel/point offsets
    color='red',
    fontweight='bold',
    ha='right',                           # Aligns text to the right edge so it stays inside the plot
    va='bottom'                           # Anchors the bottom of the text to the offset point
)
ax.legend()
plt.title('Mean tip amount by passenger count', fontsize=14)

# Create a month column
df['month'] = df['tpep_pickup_datetime'].dt.month_name()

# Create a day column
df['day'] = df['tpep_pickup_datetime'].dt.day_name()
df.head()

# Get total number of rides for each month
monthly_rides = df['month'].value_counts()

# Reorder the monthly ride list so months go in order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']
monthly_rides = monthly_rides.reindex(index=month_order)

# Show the index
monthly_rides.index

# Create a bar plot of total rides per month
plt.figure(figsize=(12,7))
ax = sns.barplot(x=monthly_rides.index, y=monthly_rides)    #julie: notice here y no need specific field name,  as it was counted by 'month' self, just dataset/series
ax.set_xticklabels(month_order)
plt.title('Ride count by month', fontsize=14)

# Repeat the above process, this time for rides by day
daily_rides = df['day'].value_counts()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_rides = daily_rides.reindex(index=day_order)

# Create bar plot for ride count by day
plt.figure(figsize=(12,7))
ax = sns.barplot(x=daily_rides.index, y=daily_rides)    #julie: notice here y no need specific field name, as it was counted by 'day' self, just dataset/series
ax.set_xticklabels(day_order)
plt.title('Ride count by day', fontsize=14)

# Repeat the process, this time for total revenue by day
total_amount_day = df.groupby(['day']).sum()[['total_amount']]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
total_amount_day = total_amount_day.reindex(index=day_order)

# Create bar plot of total revenue by day
plt.figure(figsize=(12,7))
ax = sns.barplot(x=total_amount_day.index, y=total_amount_day['total_amount'])    #julie: notice here y needs specific field name,
                                                                                #as it was sum() by this field
ax.set_xticklabels(day_order)
ax.set_ylabel('Revenue (USD)')  #label vs ticklabels, label is for the name for y axis
plt.title('Total Revenue by day', fontsize=14)

# Repeat the process, this time for total revenue by month
total_amount_month = df.groupby(['month']).sum()[['total_amount']]
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December']
total_amount_month = total_amount_month.reindex(index=month_order)

# Create a bar plot of total revenue by month
plt.figure(figsize=(12,7))
ax = sns.barplot(x=total_amount_month.index, y=total_amount_month['total_amount'])    #julie: notice here y needs specific field name,
                                                                                      #as it was sum() by this field
ax.set_xticklabels(month_order)
ax.set_ylabel('Revenue (USD)')
plt.title('Total Revenue by month', fontsize=14)

# Get number of unique drop-off location IDs
df['DOLocationID'].nunique()

# Calculate the mean trip distance for each drop-off location
mean_distance_by_dolocation = df.groupby('DOLocationID').mean()[['trip_distance']]
#mean_distance_by_dolocation
# Sort the results in descending order by mean trip distance
mean_distance_by_dolocation = mean_distance_by_dolocation.sort_values(by='trip_distance', ascending=False)

# Create a bar plot of mean trip distances by drop-off location in ascending order by distance
plt.figure(figsize=(14,6))
ax = sns.barplot(x=mean_distance_by_dolocation.index,
                 y=mean_distance_by_dolocation['trip_distance'],
                 order=mean_distance_by_dolocation.index[::-1])
ax.set_xticklabels([])
ax.set_xticks([])
plt.title('Mean trip distance by drop-off location', fontsize=14);

#BONUS CONTENT
# 1. Generate random points on a 2D plane from a normal distribution
test = np.round(np.random.normal(10, 5, (3000, 2)), 1)
midway = int(len(test)/2)  # Calculate midpoint of the array of coordinates
start = test[:midway]      # Isolate first half of array ("pick-up locations")
end = test[midway:]        # Isolate second half of array ("drop-off locations")

# 2. Calculate Euclidean distances between points in first half and second half of array
distances = (start - end)**2
distances = distances.sum(axis=-1)
distances = np.sqrt(distances)

# 3. Group the coordinates by "drop-off location", compute mean distance
test_df = pd.DataFrame({'start': [tuple(x) for x in start.tolist()],
                   'end': [tuple(x) for x in end.tolist()],
                   'distance': distances})
data = test_df[['end', 'distance']].groupby('end').mean()
data = data.sort_values(by='distance')

# 4. Plot the mean distance between each endpoint ("drop-off location") and all points it connected to
plt.figure(figsize=(14,6))
ax = sns.barplot(x=data.index,
                 y=data['distance'],
                 order=data.index)
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_xlabel('Endpoint')
ax.set_ylabel('Mean distance to all other points')
ax.set_title('Mean distance between points taken randomly from normal distribution');

# Check if all drop-off locations are consecutively numbered
df['DOLocationID'].max() - len(set(df['DOLocationID']))

plt.figure(figsize=(16,4))
# DOLocationID column is numeric, so sort in ascending order
sorted_dropoffs_location = df['DOLocationID'].sort_values()

# Convert to string
sorted_dropoffs_location = sorted_dropoffs_location.astype('str')

# Plot
sns.histplot(sorted_dropoffs_location, bins=range(0, df['DOLocationID'].max()+1, 1))
plt.xticks([])   #julie: remove xticklabels as well
plt.xlabel('Drop-off locations')
plt.title('Histogram of rides by drop-off location', fontsize=14);

df['trip_duration'] = (df['tpep_dropoff_datetime']-df['tpep_pickup_datetime'])
df.head()