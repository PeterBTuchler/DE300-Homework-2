# DE300-Homework-2
### Peter Tuchler

## How to run the code:
Put all of the files into one folder, including the Docker File, Python Code, Shell Scripts, and both csv files

In the command line, type bash HW2_BUILD.sh
Once inside of the docker container, navigate to the data folder by typing cd data
Then run the command bash RUN_PY_PYSPARK.sh
This will run all of the python code and save the results as png files and txt files

## Results

Note: All of the png and txt outputs are included in the repository.

Part 1:

## Notes on Errors
For parts 1 and 2, I was able to do all of the loading and filtering of the data using pyspark, but could not successfully join them together despite trying multiple methods. Here is the code I used:
# Load Data with Pyspark
conf = SparkConf().setAppName("CSV to RDD")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# Reading from CSV into RDD
rdd_taxi = sc.textFile("nyc_taxi_june.csv")
rdd_taxi = rdd_taxi.zipWithIndex().map(lambda row: row[0])
header_taxi = rdd_taxi.first()  # Extract the header
header_taxi = header_taxi.split(",")
rdd_taxi = rdd_taxi.map(lambda row: tuple(row.split(",")))
rdd_taxi = rdd_taxi.filter(lambda x: x != tuple(header_taxi))

# Step 1: Filter Taxi Data
min_dur = 3*60
max_dur = 6*3600
min_long = -74.03
max_long = -73.75
min_lat = 40.63
max_lat = 40.85
filtered_rdd_taxi1 = rdd_taxi.filter(lambda row: float(row[10]) >= min_dur and float(row[10]) <= max_dur)
filtered_rdd_taxi2 = filtered_rdd_taxi1.filter(lambda row: float(row[5]) != float(row[7]) or float(row[6]) != float(row[8]))
filtered_rdd_taxi3 = filtered_rdd_taxi2.filter(lambda row: float(row[5]) >= min_long and float(row[5]) <= max_long and float(row[7]) >= min_long and float(row[7]) <= max_long and float(row[6]) >= min_lat and float(row[6]) <= max_lat and float(row[8]) >= min_lat and float(row[8]) <= max_lat)

# Step 2: Add Weather Attributes
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

