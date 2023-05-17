# Parameters:

# !pip install pyspark                    # Just for google colab
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pyspark
from pyspark.mllib.clustering import KMeans as KMeans_for_RDD          #Giving it a different name
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
import math
import datetime
from pyspark.ml.clustering import KMeans


# # Below is Pyspark code that mostly works except for one small error towards the end
# # See attached screenshots for details of the error. Numpy is used instead
# # Load Data with Pyspark
# conf = SparkConf().setAppName("CSV to RDD")
# sc = SparkContext(conf=conf)
# spark = SparkSession.builder.getOrCreate()

# # Reading from CSV into RDD
# rdd_taxi = sc.textFile("nyc_taxi_june.csv")
# rdd_taxi = rdd_taxi.zipWithIndex().map(lambda row: row[0])
# header_taxi = rdd_taxi.first()  # Extract the header
# header_taxi = header_taxi.split(",")
# rdd_taxi = rdd_taxi.map(lambda row: tuple(row.split(",")))
# rdd_taxi = rdd_taxi.filter(lambda x: x != tuple(header_taxi))

# # Step 1: Filter Taxi Data
# min_dur = 3*60
# max_dur = 6*3600
# min_long = -74.03
# max_long = -73.75
# min_lat = 40.63
# max_lat = 40.85
# filtered_rdd_taxi1 = rdd_taxi.filter(lambda row: float(row[10]) >= min_dur and float(row[10]) <= max_dur)
# filtered_rdd_taxi2 = filtered_rdd_taxi1.filter(lambda row: float(row[5]) != float(row[7]) or float(row[6]) != float(row[8]))
# filtered_rdd_taxi3 = filtered_rdd_taxi2.filter(lambda row: float(row[5]) >= min_long and float(row[5]) <= max_long and float(row[7]) >= min_long and float(row[7]) <= max_long and float(row[6]) >= min_lat and float(row[6]) <= max_lat and float(row[8]) >= min_lat and float(row[8]) <= max_lat)

# # Step 2: Add Weather Attributes
# rdd_weather = sc.textFile("weather_hourly_june.csv")
# rdd_weather = rdd_weather.zipWithIndex().map(lambda row: row[0])
# header_weather = rdd_weather.first()  # Extract the header
# header_weather = header_weather.split(",")
# rdd_weather = rdd_weather.map(lambda row: tuple(row.split(",")))
# rdd_weather = rdd_weather.filter(lambda x: x != tuple(header_weather))

# def format_date(str):
#   date = str[:11]
#   time = str[11:]
#   return f"{int(date[5:7])}/{int(date[8:10])}/{date[0:4]} {int(time[0:2])}:00"

# rdd_taxi_date = filtered_rdd_taxi3.map(lambda x: (format_date(x[2]), x[0], x[1], format_date(x[3]), x[4], x[5], x[6], x[7], x[8], x[9], x[10]))
# rdd_weather_small = rdd_weather.map(lambda x: (x[0], x[1], x[5], x[11]))
# print(rdd_taxi_date.collect()[0:10])
# print(rdd_weather_small.collect()[0:10])
# # Join columns of rdd
# # rdd = rdd_taxi_date.leftOuterJoin(rdd_weather_small)

# # print(rdd.collect()[0:10])
# # taxi_header = sc.parallelize([('pickup_datetime', 'id', 'vendor_id', 'dropoff_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration')])  # Header row as a tuple
# # rdd_taxi_date = taxi_header.union(rdd_taxi_date)

# # weather_header = sc.parallelize([('pickup_datetime', 'Temp', 'Pressure', 'Precip')])  # Header row as a tuple
# # rdd_weather_small = weather_header.union(rdd_weather_small)

# df1 = spark.createDataFrame(rdd_taxi_date)
# df1.show()
# df2 = spark.createDataFrame(rdd_weather_small)
# df2.show()

# # Perform a join using DataFrames
# joined_df = df1.join(df2, on='_1', how='left')

# # Show the joined DataFrame
# joined_df.show()

# rdd = joined_df.toRDD()

# print(rdd.collect()[0:10])


# Load Data
taxi_data = pd.read_csv(r'nyc_taxi_june.csv')
df_taxi = pd.DataFrame(taxi_data)
taxi = df_taxi.values                         # converts to numpy representation

weather_data = pd.read_csv(r'weather_hourly_june.csv')
df_weather = pd.DataFrame(weather_data)
weather = df_weather.values                   # converts to numpy representation


# Step 1: Filter Taxi Data
trips_to_remove = []
index = 0

for trip in taxi:
  if int(trip[10]) < 3*60 or int(trip[10]) > 6*3600:
    trips_to_remove.append(index)
  elif trip[5] == trip [7] and trip[6] == trip [8]:
    trips_to_remove.append(index)
  elif trip[5] < -74.03 or trip[5] > -73.75 or trip[6] < 40.63 or trip[6] > 40.85 or trip[7] < -74.03 or trip[7] > -73.75 or trip[8] < 40.63 or trip[8] > 40.85:
    trips_to_remove.append(index)
  index += 1
# print(len(trips_to_remove))
# # print(trips_to_remove)

taxi = np.delete(taxi,trips_to_remove,0)   #removes specified trips from dataset

# Step 2: Add Weather Attributes
temp = []
pressure = []
precipitation = []

for trip in taxi:
  date = trip[2][:11]
  time = trip[2][11:]
  date_time = f"{int(date[5:7])}/{int(date[8:10])}/{date[0:4]} {int(time[0:2])}:00"
  weather_index = np.where(weather == date_time)[0][0]
  temp.append(weather[weather_index][1])
  pressure.append(weather[weather_index][5])
  precipitation.append(weather[weather_index][10])

# Adding weather columns to taxi data
joined = np.hstack((taxi, np.atleast_2d(temp).T))
joined = np.hstack((joined, np.atleast_2d(pressure).T))
joined = np.hstack((joined, np.atleast_2d(precipitation).T))

# Step 3: Clustering with pyspark
spark = SparkSession.builder.appName("NumPy to RDD").getOrCreate()
sc = spark.sparkContext
temp_tuples = [tuple(row) for row in joined]

rdd = sc.parallelize(temp_tuples)

# Kmeans Clustering
rdd_clust = rdd.map(lambda x: (x[5], x[6]))
clusters = KMeans_for_RDD.train(rdd_clust, k=3, maxIterations = 10, initializationMode = "random")


predictions = clusters.predict(rdd_clust)

clustered_data = rdd.zip(predictions).map(lambda x: (x[0][0], x[0][1], x[1]))

x = rdd.map(lambda x: x[5]).collect()         # x coordinates
y = rdd.map(lambda y: y[6]).collect()         # y coordinates

colors_pick = clustered_data.map(lambda x: x[2]).collect()

fig1 = plt.figure()
plt.scatter(x, y, c = colors_pick)
plt.plot([-73.87, -73.79], [40.78, 40.64], marker='*', ls='none', ms=20, color = 'r')            # Plots LaGuardia and JFK
plt.text(-73.87,40.78,'JFK')
plt.text(-73.79,40.64,'LGA')
plt.title("Pickup Clusters")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig('pickup_clusters.png')
plt.show()
plt.clf()

# Silhouette Score Calculation
rdd_clust_df = rdd_clust.toDF(['long', 'lat'])
assembler_clust = VectorAssembler(inputCols=['long', 'lat'], outputCol = "features")
data_clust_assembled = assembler_clust.transform(rdd_clust_df)          # This is now a vector

evaluator = ClusteringEvaluator()

file = open("Silhouette_Scores.txt", 'w')
for number in [2, 3, 4, 5]:
  kmeans = KMeans(k=number, seed = 1234)
  model = kmeans.fit(data_clust_assembled)
  predictions = model.transform(data_clust_assembled)
  score = evaluator.evaluate(predictions)
  file.write(f"{number} : {score}\n")
  # print(f"{number} : {score}")
file.close()

# Silhouette Score for 3 Clusters is 0.92
# Silhouette Score for 3 Clusters is 0.58
# Silhouette Score for 4 Clusters is 0.65
# Silhouette Score for 5 Clusters is 0.60

# 2 clusters is visually too few clusters and is not useful for analysis. 3, 4, and 5 have very similar silhouette values,
# so I have chosen a value of 3 because it seems to match the real world regional split: Lower manhatten & Brooklin, Upper manhatten & The Bronx,
# and the sparser region of Queens

# Part C
# JFK Coordinates: 40.64, -73.79
# LaGuardia Coordinates: 40.78, -73.87
# JFK is way to the bottom right in the sparse purple cluster
# LaGuardia is near the intersection of purpla and green clusters (with 3 clusters)

# Park D
x_drop = rdd.map(lambda x: x[7]).collect()         # x coordinates
y_drop = rdd.map(lambda y: y[8]).collect()         # y coordinates

colors_pick = clustered_data.map(lambda x: x[2]).collect()

fig2 = plt.figure()
plt.scatter(x_drop, y_drop, c = colors_pick)
plt.plot([-73.87, -73.79], [40.78, 40.64], marker='*', ls='none', ms=20, color = 'r')            # Plots LaGuardia and JFK
plt.title("Dropoff Location with Pickup-based Clustering Classification")
plt.text(-73.87,40.78,'JFK')
plt.text(-73.79,40.64,'LGA')
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig('Dropoff_of_pickup_clusters.png')
plt.show()
plt.clf()

#Using the cluster centriods from pickup model, assign dropoff points to clusters
rdd_clust_drop = rdd.map(lambda x: (x[7], x[8]))
predictions_drop = clusters.predict(rdd_clust_drop)

clustered_data_drop = rdd.zip(predictions_drop).map(lambda x: (x[0][0], x[0][1], x[1]))

colors_drop = clustered_data_drop.map(lambda x: x[2]).collect()

fig3 = plt.figure()
plt.scatter(x_drop, y_drop, c = colors_drop)
plt.plot([-73.87, -73.79], [40.78, 40.64], marker='*', ls='none', ms=20, color = 'r')            # Plots LaGuardia and JFK
plt.text(-73.87,40.78,'JFK')
plt.text(-73.79,40.64,'LGA')
plt.title("Dropoff Clusters (pickup centroids)")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.savefig('Dropoffs_clusters.png')
plt.show()
plt.clf()

# Silhouette Score Calculation
rdd_clust_df_drop = rdd_clust_drop.toDF(['long', 'lat'])
assembler_clust_drop = VectorAssembler(inputCols=['long', 'lat'], outputCol = "features")
data_clust_assembled_drop = assembler_clust_drop.transform(rdd_clust_df_drop)          # This is now a vector

evaluator = ClusteringEvaluator()
kmeans_drop = KMeans(k=3, seed = 1234)
model_drop = kmeans_drop.fit(data_clust_assembled_drop)
predictions_drop = model_drop.transform(data_clust_assembled_drop)
score_drop = evaluator.evaluate(predictions_drop)

file2 = open("Silhouette_Scores_Final_Value.txt", 'w')
# print(score_drop)
file2.write(str(score))
file2.close()

# With 3 clusters, clustering on dropoff, we get a silhouette score of 0.585

# 4: Analysing Intercluster and Intracluster lists

clust_labels = list(zip(colors_pick, colors_drop))

rdd_labels = sc.parallelize(clust_labels)


rdd_count = rdd_labels.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda a,b: a+b)
clust_counts = dict(rdd_count.collect())
 

count_00 = clust_counts[(0,0)]
count_01 = clust_counts[(0,1)] + clust_counts[(1,0)]
count_02 = clust_counts[(0,2)] + clust_counts[(2,0)]
count_11 = clust_counts[(1,1)]
count_12 = clust_counts[(1,2)] + clust_counts[(2,1)]
count_22 = clust_counts[(2,2)]

file3 = open("Intra_Inter_Cluster_Trips.txt", 'w')
file3.write(f'0-0 : {count_00}\n0-1: {count_01}\n0-2 : {count_02}\n1-1 : {count_11}\n1-2 : {count_12}\n2-2 : {count_22}\n')
# print(f'{count_00}, {count_01}, {count_02}, {count_11}, {count_12}, {count_22}')
# print(count_00 + count_01 + count_02 + count_11 + count_12 + count_22)
file3.close()

# As we can see, the cluster with the most inter cluster trips is 1 with 25511, followed by 2 with 23752, followed by 0 with 671

# The pair with most between cluster travel is 1-2 with 22212, followed by 0-1 with 3105, followed by 0-2 with 2687

# 5: Histogram of (log) trip durations

durations = rdd.map(lambda x: x[10]).collect()
# print(durations)
log_durations = np.log(durations)

fig4 = plt.figure()
plt.hist(log_durations, bins=30, edgecolor='black')
plt.xlabel('log_durations')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.savefig('Log_Duration_Histogram.png')
plt.show()
plt.clf()


# I see that the log of the durations it is farily normally distributed with a peak around e^(6.6) = 735 seconds. 
# There is also a small amount of right skew

# Average of log trip duration for each day

rdd_day = rdd.map(lambda x: (int(x[2][8:10]), (np.log(float(x[10])), 1)))                         #2 digit day (key) and the duration in seconds (value)

rdd_day_sum = rdd_day.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
rdd_day_avg = rdd_day_sum.map(lambda x: (x[0], x[1][0] / x[1][1]))
rdd_day_avg_sort = rdd_day_avg.sortByKey(ascending = True)

day = rdd_day_avg_sort.map(lambda x: x[1]).collect()
log_dur = rdd_day_avg_sort.map(lambda x: x[0]).collect()

fig5 = plt.figure()
plt.plot(log_dur, day)
plt.title("Average trip (log) for each day in June")
plt.xlabel("Day in June")
plt.ylabel("Log of duration in seconds")
plt.savefig('Log_Durations_Day_of_June.png')
plt.show()
plt.clf()

# Part 6: durations by time periods

# Day of week:
# June 1 2016 was a wednesday

def day_of_week(day):
  if day == 0:
    return("Tuesday")
  elif day == 1:
    return("Wednesday")
  elif day == 2:
    return("Thursday")
  elif day == 3:
    return("Friday")
  elif day == 4:
    return("Saturday")
  elif day == 5:
    return("Sunday")
  else:
    return("Monday")

rdd_DoW = rdd.map(lambda x: (int(x[2][8:10]) % 7, (np.log(float(x[10])), 1)))               # DoW means day of week. 1 means wednesday, 2 means Thursday... 0 means Tuesday

rdd_DoW_sum = rdd_DoW.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
rdd_DoW_avg = rdd_DoW_sum.map(lambda x: (x[0], x[1][0] / x[1][1]))
rdd_DoW_avg_sort = rdd_DoW_avg.sortByKey(ascending = True)

day_of_the_week = rdd_DoW_avg_sort.map(lambda x: day_of_week(x[0])).collect()
log_dur = rdd_DoW_avg_sort.map(lambda x: x[1]).collect()

fig6 = plt.figure()
plt.plot(day_of_the_week, log_dur)
plt.title("Average trip (log) for each day of the week")
plt.xlabel("Day of Week")
plt.ylabel("Log of duration in seconds")
plt.savefig('Log_Duration_Day_of_Week.png')
plt.show()
plt.clf()

# Morning, Afternoon, Evening, or Night
# Morning is 7-12, Afternoon is 12-17, Evening is 17-0, night is 0-7

# function for part of the day:
def part_of_day(t):
  if t<7:
    return("night")
  elif t<12:
    return("morning")
  elif t<17:
    return("afternoon")
  else:
    return("evening")

rdd_PoD = rdd.map(lambda x: (part_of_day(int(x[2][11:13])), (np.log(float(x[10])), 1)))               # PoD means part of day.

rdd_PoD_sum = rdd_PoD.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
rdd_PoD_avg = rdd_PoD_sum.map(lambda x: (x[0], x[1][0] / x[1][1]))
rdd_PoD_avg_sort = rdd_PoD_avg.sortByKey(ascending = True)

part_of_the_day= rdd_PoD_avg_sort.map(lambda x: x[0]).collect()
log_dur = rdd_PoD_avg_sort.map(lambda x: x[1]).collect()

fig7 = plt.figure()
plt.plot(part_of_the_day, log_dur)
plt.title("Average trip (log) for each part of the day")
plt.xlabel("Part of the day")
plt.ylabel("Log of duration in seconds")
plt.savefig('Log_Duration_Part_of_Day.png')
plt.show()
plt.clf()

# Weekend vs weekday
def weekend_or_weekday(t):
  if t == 4 or t == 5:
    return("weekend")
  else:
    return("weekday")

rdd_WoW = rdd.map(lambda x: (weekend_or_weekday(int(x[2][8:10])), (np.log(float(x[10])), 1)))               # WoW means weekday or weekend

rdd_WoW_sum = rdd_WoW.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
rdd_WoW_avg = rdd_WoW_sum.map(lambda x: (x[0], x[1][0] / x[1][1]))
rdd_WoW_avg_sort = rdd_WoW_avg.sortByKey(ascending = True)

end_or_day= rdd_WoW_avg_sort.map(lambda x: x[0]).collect()
log_dur = rdd_WoW_avg_sort.map(lambda x: x[1]).collect()

fig8 = plt.figure()
plt.bar(end_or_day, log_dur)
plt.title("Average trip (log) for weekend or weekday")
plt.ylim(bottom = 6.2)
plt.xlabel("Weekend or weekday")
plt.ylabel("Log of duration in seconds")
plt.savefig('Log_Duration_Weekend.png')
plt.show()
plt.clf()

# Weekday rush hour or not
def weekday_rush_hour(hour, day):
  if day != 4 and day != 5:                             #A weekday
    if (hour >= 8 and hour <= 9) or (hour >= 15 and hour <= 19):
      return("Rush Hour")
    else:
      return("Normal")
  else:
    return("Normal")

rdd_RH = rdd.map(lambda x: (weekday_rush_hour(int(x[2][11:13]), int(x[2][8:10])), (np.log(float(x[10])), 1)))               # RH means rush hour

rdd_RH_sum = rdd_RH.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
rdd_RH_avg = rdd_RH_sum.map(lambda x: (x[0], x[1][0] / x[1][1]))

rush_hour= rdd_RH_avg.map(lambda x: x[0]).collect()
log_dur = rdd_RH_avg.map(lambda x: x[1]).collect()

fig9 = plt.figure()
plt.bar(rush_hour, log_dur)
plt.title("Average trip (log) for Weekday Rush Hour or Normal times")
plt.ylim(bottom = 6.2)
plt.xlabel("Rush hour or Normal")
plt.ylabel("Log of duration in seconds")
plt.savefig('Log_Duration_Rush_Hour.png')
plt.show()
plt.clf()

# Part 7:
def great_circle(long_p, lat_p, long_d, lat_d):
  long_p_rad = math.radians(long_p)
  lat_p_rad = math.radians(lat_p)
  long_d_rad = math.radians(long_d)
  lat_d_rad = math.radians(lat_d)
 
  dlong = long_d_rad - long_p_rad
  dlat = lat_d_rad - lat_p_rad
  a = math.sin(dlat/2)**2 + math.cos(lat_p_rad) * math.cos(lat_d_rad) * math.sin(dlong/2)**2
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
  radius = 6371 

  distance = radius * c
  return distance

rdd_Dist = rdd.map(lambda x: (np.log(great_circle(x[5], x[6], x[7], x[8])), np.log(float(x[10]))))
# print(rdd_Dist.collect()[0:10])

log_distance = rdd_Dist.map(lambda x: x[0]).collect()
log_dur = rdd_Dist.map(lambda x: x[1]).collect()

fig10 = plt.figure()
plt.scatter(log_distance, log_dur, s = 4)
plt.title("Log of the distance vs log of the duration")
plt.xlabel("Log of the distance in km")
plt.ylabel("Log of duration in seconds")
plt.savefig('Log_Distance_vs_Log_Duration.png')
plt.show()
plt.clf()

# what is the trasformation? We see that it is log linear

# Part 8
# First add in all new features to the (i.e. great circle distance, day of month, day of week, part of week, weekday rush hour)

big_rdd = rdd.map(lambda x: (x[4], x[5], x[6], x[7], x[8], x[10], x[11], x[12], x[13], int(x[2][8:10]), day_of_week(int(x[2][8:10]) % 7), part_of_day(int(x[2][11:13])), weekend_or_weekday(int(x[2][8:10])), weekday_rush_hour(int(x[2][11:13]), int(x[2][8:10])), float(np.log(great_circle(x[5], x[6], x[7], x[8]))))).collect()

# print(big_rdd[0:5])

# Regression
# Create data frame from our spark session called spark
df = spark.createDataFrame(big_rdd, ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_duration', 'temperature', 'pressure', 'precipitation', 'day_of_month', 'day_of_week', 'part_of_day', 'weekend_or_weekday', 'weekday_rush_hour', 'log_great_circle'])

# df.show()

# One Hot Encoding -------------------------------------------------------------------------------------VVVV
# Apply string indexing to the categorical column  -->  'day_of_month'
string_indexer = StringIndexer(inputCol='day_of_month', outputCol='indexed_day_of_month')
indexed_df = string_indexer.fit(df).transform(df)

# Apply one-hot encoding to the indexed column
one_hot_encoder = OneHotEncoder(inputCols=['indexed_day_of_month'], outputCols=['encoded_day_of_month'])
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)



# Apply string indexing to the categorical column  -->  'day_of_week'
string_indexer = StringIndexer(inputCol='day_of_week', outputCol='indexed_day_of_week')
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Apply one-hot encoding to the indexed column
one_hot_encoder = OneHotEncoder(inputCols=['indexed_day_of_week'], outputCols=['encoded_day_of_week'])
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)



# Apply string indexing to the categorical column  -->  'part_of_day'
string_indexer = StringIndexer(inputCol='part_of_day', outputCol='indexed_part_of_day')
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Apply one-hot encoding to the indexed column
one_hot_encoder = OneHotEncoder(inputCols=['indexed_part_of_day'], outputCols=['encoded_part_of_day'])
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)



 # Apply string indexing to the categorical column  -->  'weekend_or_weekday'
string_indexer = StringIndexer(inputCol='weekend_or_weekday', outputCol='indexed_weekend_or_weekday')
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Apply one-hot encoding to the indexed column
one_hot_encoder = OneHotEncoder(inputCols=['indexed_weekend_or_weekday'], outputCols=['encoded_weekend_or_weekday'])
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)



# Apply string indexing to the categorical column  -->  'weekday_rush_hour'
string_indexer = StringIndexer(inputCol='weekday_rush_hour', outputCol='indexed_weekday_rush_hour')
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Apply one-hot encoding to the indexed column
one_hot_encoder = OneHotEncoder(inputCols=['indexed_weekday_rush_hour'], outputCols=['encoded_weekday_rush_hour'])
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)



# Apply string indexing to the categorical column  -->  'log_great_circle'
string_indexer = StringIndexer(inputCol='log_great_circle', outputCol='indexed_log_great_circle')
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Apply one-hot encoding to the indexed column
one_hot_encoder = OneHotEncoder(inputCols=['indexed_log_great_circle'], outputCols=['encoded_log_great_circle'])
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)
#-------------------------------------------------------------------------------------------------------------------^^^^

# encoded_df.show()

# Assemble the features into a vector column
assembler = VectorAssembler(inputCols=['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'temperature', 'pressure', 'precipitation', 'encoded_day_of_month', 'encoded_day_of_week', 'encoded_part_of_day', 'encoded_weekend_or_weekday', 'encoded_weekday_rush_hour', 'encoded_log_great_circle'], outputCol="features")
assembled_df = assembler.transform(encoded_df)

# Split data into testing and training
train_data = assembled_df.filter(assembled_df.day_of_month < 24)
test_data = assembled_df.filter(assembled_df.day_of_month >= 24)

drop_features = ['day_of_month', 'day_of_week', 'part_of_day', 'weekend_or_weekday', 'weekday_rush_hour', 'log_great_circle', 'features']

for feature in drop_features:
  train_data = train_data.drop(feature)
  test_data = test_data.drop(feature)

# Reassemble
assembler2 = VectorAssembler(inputCols=['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'temperature', 'pressure', 'precipitation', 'encoded_day_of_week', 'encoded_part_of_day', 'encoded_weekend_or_weekday', 'encoded_weekday_rush_hour'], outputCol="features")
assembled_df = assembler2.transform(encoded_df)

# assembled_df.show()

train_assembled_df = assembler2.transform(train_data)
test_assembled_df = assembler2.transform(test_data)

# Create a LinearRegression instance
lr = LinearRegression(featuresCol="features", labelCol="trip_duration")

# Fit the model to the training data
lr_model = lr.fit(train_assembled_df)

# Make predictions on the test data
predictions = lr_model.transform(test_assembled_df)

# Convert the 'predictions' DataFrame to a Pandas DataFrame
pandas_df = predictions.select("features", "trip_duration", "prediction").toPandas()

# Get the string representation of the Pandas DataFrame
output_string = pandas_df.to_string(index=False)

# You can save the output to a file or manipulate it further as needed
file6 = open("Regression_Predictions.txt", "w")
file6.write(output_string[0:3406])
file6.close()

# Create a RegressionEvaluator instance
evaluator = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="r2")
evaluator2 = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="mse")
evaluator3 = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="mae")

# Compute the MSE
mse = evaluator2.evaluate(predictions)

# Compute the R-squared
r2 = evaluator.evaluate(predictions)

# Compute the R-squared
mae = evaluator3.evaluate(predictions)

# Print the MSE and R-squared
file4 = open("Regression_Results.txt", 'w')
file4.write(f'Mean Squared Error (MSE): {mse}\n')
file4.write(f'R-squared (R^2): {r2}\n')
file4.write(f'MAE: {mae}\n')
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R-squared (R^2): {r2}")
# print(f"MAE: {mae}")
file4.close()

# Random Forest

num_trees = 50
max_depth = 10

rf = RandomForestRegressor(featuresCol="features", labelCol="trip_duration", numTrees = num_trees, maxDepth = max_depth)

rf_model = rf.fit(train_assembled_df)


RF_predictions = rf_model.transform(test_assembled_df)
# RF_predictions.select("features", "trip_duration", "prediction").show()

# Convert the 'predictions' DataFrame to a Pandas DataFrame
RF_pandas_df = RF_predictions.select("features", "trip_duration", "prediction").toPandas()

# Get the string representation of the Pandas DataFrame
RF_output_string = RF_pandas_df.to_string(index=False)

# You can save the output to a file or manipulate it further as needed
file7 = open("Random_Forest_Predictions.txt", "w")
file7.write(RF_output_string[0:3406])
file7.close()

RF_evaluator = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="r2")
RF_evaluator2 = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="rmse")
RF_evaluator3 = RegressionEvaluator(labelCol="trip_duration", predictionCol="prediction", metricName="mae")

rmse = RF_evaluator2.evaluate(RF_predictions)
r2 = RF_evaluator.evaluate(RF_predictions)
mae = RF_evaluator3.evaluate(RF_predictions)

# Print the MSE and R-squared
file5 = open("Random_Forest_Results.txt", 'w')
file5.write(f'Root Mean Squared Error (MSE): {rmse}\n')
file5.write(f'R-squared (R^2): {r2}\n')
file5.write(f'MAE: {mae}\n')
# print(f"Root Mean Squared Error (MSE): {rmse}")
# print(f"R-squared (R^2): {r2}")
# print(f"MAE: {mae}")
file5.close()

# These are the results:
# Root Mean Squared Error (MSE): 439.3237732670487
# R-squared (R^2): 0.5681696394740351
# MAE: 324.454834780395

