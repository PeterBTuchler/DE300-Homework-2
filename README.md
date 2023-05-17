# DE300-Homework-2
### Peter Tuchler

## How to run the code:
Put all of the files into one folder, including the Docker File, Python Code, Shell Scripts, and both csv files

In the command line, type bash HW2_BUILD.sh
Once inside of the docker container, navigate to the data folder by typing cd data
Then run the command bash RUN_PY_PYSPARK.sh
This will run all of the python code and save the results as png files and txt files

## Results

Note: All of the png and txt outputs are included in the repository for viewing purposes.

Part 1:
I filtered the data using the pyspark function filter() and logic statements

Part 2: 
I used a left join to join the weather to the taxi info



Part 3: 
I tried K values of 2, 3, 4, and 5, and found that 3 clusters is the optimal choice.
The silhouette scores for each of these number of clusters are:

2 : 0.9155878433582075

3 : 0.8437322530008426

4 : 0.6572225736103711

5 : 0.6463182057652361

Clearly 2 and 3 clusters have the best scores, but I went with 3 because 2 clusters is visually too few clusters and is not useful for analysis so I have chosen a value of 3 because it seems to match the real world regional split: Lower manhatten & Brooklin, Upper manhatten & The Bronx, and the sparser region of Queens

For my final model has a silhouette score of 0.8437322530008426.

JFK and LaGuardia are both in the purple cluster, though JFK is very near by the Purple Cluster and LaGuardia is at the far edge of the Purple Custer. (See pickup_clusters.png)

For the reassigned clusters based on dropoff coordinates (but the same original centroids), see Dropoffs_clusters.png. We got a new silhouette score of 0.6463182057652361.

Part 4:
The results are output to Intra_Inter_Cluster_Trips.txt

Decreasing order of Intra Cluster Trips

0-0 : 25717

2-2 : 23513

1-1 : 671


Decreasing order of Inter Cluster Trips

0-2 : 22245

0-1: 3131

1-2 : 2661

As we can see, the cluster with the most inter cluster trips is 1 with 25511, followed by 2 with 23752, followed by 0 with 671. 
The pair with most between cluster travel is 1-2 with 22212, followed by 0-1 with 3105, followed by 0-2 with 2687

Part 5: 
All graphs can be viewed in their respectively named png files


I see that the log of the durations it is farily normally distributed with a peak around e^(6.6) = 735 seconds. 
There is also a small amount of right skew

For the average trip during each day of June, we see that there are 4 evenly spaced dips, meaning that this is cyclical on the week level.

Part 6:
All graphs can be viewed in their respectively named png files

In general, we confirm the weekly cycle trip durations. Week days have longer trip durations on average. Rush Hour has longer trip durations on average. Afternoon has the longest trips, followed by Morning, then evening, then night.

Part 7:
In the plot Log_Distance_vs_Log_Duration, we see that it is mostly linear with around a slope of  about 1.5. The appropriate transformation used is the log tansformation, because if we find the slope of a log-log plot, we know the power of the relationship between distance and time. There also seems to be an envelope where no points are to the bottom right. This is likely due to the speed limit on roads.

Part 8:
The resuts of the regression models (both predictions and RMSE, MAE, and R2) are in the attached text files of corresponding name. The most important features for the random forest model were the great circle distance, whether it is a weekend, and precipitation.

## Notes on avoided Errors
For parts 1 and 2, I was able to do all of the loading and filtering of the data using pyspark, but could not successfully join them together despite trying multiple methods. Here is the code I used:
The code is also included in my python file, but is commented out.

/ Load Data with Pyspark
conf = SparkConf().setAppName("CSV to RDD")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

/ Reading from CSV into RDD
rdd_taxi = sc.textFile("nyc_taxi_june.csv")
rdd_taxi = rdd_taxi.zipWithIndex().map(lambda row: row[0])
header_taxi = rdd_taxi.first()  # Extract the header
header_taxi = header_taxi.split(",")
rdd_taxi = rdd_taxi.map(lambda row: tuple(row.split(",")))
rdd_taxi = rdd_taxi.filter(lambda x: x != tuple(header_taxi))

/ Step 1: Filter Taxi Data
min_dur = 3*60
max_dur = 6*3600
min_long = -74.03
max_long = -73.75
min_lat = 40.63
max_lat = 40.85
filtered_rdd_taxi1 = rdd_taxi.filter(lambda row: float(row[10]) >= min_dur and float(row[10]) <= max_dur)
filtered_rdd_taxi2 = filtered_rdd_taxi1.filter(lambda row: float(row[5]) != float(row[7]) or float(row[6]) != float(row[8]))
filtered_rdd_taxi3 = filtered_rdd_taxi2.filter(lambda row: float(row[5]) >= min_long and float(row[5]) <= max_long and float(row[7]) >= min_long and float(row[7]) <= max_long and float(row[6]) >= min_lat and float(row[6]) <= max_lat and float(row[8]) >= min_lat and float(row[8]) <= max_lat)

/ Step 2: Add Weather Attributes
rdd_weather = sc.textFile("weather_hourly_june.csv")
rdd_weather = rdd_weather.zipWithIndex().map(lambda row: row[0])
header_weather = rdd_weather.first()  # Extract the header
header_weather = header_weather.split(",")
rdd_weather = rdd_weather.map(lambda row: tuple(row.split(",")))
rdd_weather = rdd_weather.filter(lambda x: x != tuple(header_weather))

def format_date(str):
  date = str[:11]
  time = str[11:]
  return f"{int(date[5:7])}/{int(date[8:10])}/{date[0:4]} {int(time[0:2])}:00"

rdd_taxi_date = filtered_rdd_taxi3.map(lambda x: (format_date(x[2]), x[0], x[1], format_date(x[3]), x[4], x[5], x[6], x[7], x[8], x[9], x[10]))
rdd_weather_small = rdd_weather.map(lambda x: (x[0], x[1], x[5], x[11]))
print(rdd_taxi_date.collect()[0:10])
print(rdd_weather_small.collect()[0:10])
/ Join columns of rdd
/ rdd = rdd_taxi_date.leftOuterJoin(rdd_weather_small)

/ print(rdd.collect()[0:10])
/ taxi_header = sc.parallelize([('pickup_datetime', 'id', 'vendor_id', 'dropoff_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration')])  # Header row as a tuple
/ rdd_taxi_date = taxi_header.union(rdd_taxi_date)

/ weather_header = sc.parallelize([('pickup_datetime', 'Temp', 'Pressure', 'Precip')])  # Header row as a tuple
/ rdd_weather_small = weather_header.union(rdd_weather_small)

df1 = spark.createDataFrame(rdd_taxi_date)
df1.show()
df2 = spark.createDataFrame(rdd_weather_small)
df2.show()

/ Perform a join using DataFrames
joined_df = df1.join(df2, on='_1', how='left')

/ Show the joined DataFrame
joined_df.show()

rdd = joined_df.toRDD()

print(rdd.collect()[0:10])

I have included a screenshot of the error in the repository. Instead I used numpy to join the data, but please see above that the code is almost fully correct.

Also, random forest made an error unless I used less than around 50 trees, so I used 50 trees. I have attached a screenshot of the error

