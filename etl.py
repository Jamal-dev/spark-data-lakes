import configparser
from datetime import datetime
import os
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.functions import hour, weekofyear, date_format
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.types import StringType, IntegerType, LongType, TimestampType
from path import Path
import glob

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def extract_cols(col_names: List,
                 drop_dup_cols: List,
                 df,
                 partitionBy: List = None,
                 table_name: str = '',
                 if_use_select_expr: bool = False):
    """
        It extracts columns from dataframe df
        args:
         col_names:List      : List of columns that are needed to be extracted
         drop_dup_cols:List  : List of columns that are needed to be unique
         df                  : Data frame
         partitionBy:List=None: List of columns by which the table is partioned
         table_name:str      : Name of the output table that needed to
                               be saved in the paraquet format
         if_use_select_expr  : If true it will run selectExpr command
                               otherwise select command of SQL
    """
    # extract columns to create table
    if if_use_select_expr:
        table = df.selectExpr(*col_names).dropDuplicates(drop_dup_cols)

    else:
        table = df.select(*col_names).dropDuplicates(drop_dup_cols)
    if partitionBy is not None:
        # write table to parquet files partitioned by partioinBy column list
        table.write.parquet(Path(output_data) / Path(table_name + ".parquet"),
                            partitionBy=partitionBy,
                            mode="overwrite")
    else:
        # write table to parquet files
        table.write.parquet(Path(output_data) / Path(table_name + ".parquet"),
                            mode="overwrite")
    return table


def process_song_data(spark, input_data, output_data):
    """
        This procedure extracts metadata of songs stored in S3 in JSON format.
        It prepares data frames for artists and songs table from this dataset.
        It saves these tables back to S3 in Parquet file format.
        Songs table is partitioned by year of song and artist ID
        while storing as Parquet file.
        args:
         spark             :    instance of Spark context
         input_data        :    file path of directory where datasets
                                for songs is stored
         output_data       :    file path of directory where the dimensions
                                tables are to be stored as Parquet format
    """
    # get filepath to song data file
    songPath = Path(input_data) / Path("song-data/**/*.json")
    song_data = glob.glob(songPath, recursive=True)
    # schema for song dataset
    song_schema = StructType([
        StructField("artist_id", StringType()),
        StructField("artist_latitude", DoubleType()),
        StructField("artist_location", StringType()),
        StructField("artist_longitude", DoubleType()),
        StructField("artist_name", StringType()),
        StructField("duration", DoubleType()),
        StructField("num_songs", IntegerType()),
        StructField("song_id", StringType()),
        StructField("title", StringType()),
        StructField("year", IntegerType())
    ])

    # read song data file
    df = spark.read.json(song_data, song_schema)

    # songs table columns
    song_cols = ['song_id',
                 'title',
                 'artist_id',
                 'year',
                 'duration']

    # songs table
    songs_table = extract_cols(col_names=song_cols,
                               drop_dup_cols=["song_id"],
                               df=df,
                               partitionBy=["year", "artist_id"],
                               table_name='songs')
    # artists table columns
    artist_cols = ["artist_id",
                   "artist_name as name",
                   "artist_location as location",
                   "artist_latitude as lattitude",
                   "artist_longitude as longitude"]
    # artists table
    artists_table = extract_cols(col_names=artist_cols,
                                 drop_dup_cols=["artist_id"],
                                 df=df,
                                 partitionBy=None,
                                 table_name='artists',
                                 if_use_select_expr=True)


def process_log_data(spark, input_data, output_data):
    """
    Load JSON input data (log_data) from input_data path,
    process the data to extract users_table, time_table,
    songplays_table, and store the queried data to parquet files.
    args:
     spark             :    instance of Spark context
     input_data        :    file path of directory where datasets
                            for user activity log is stored
     output_data       :    file path of directory where the dimensions
                            tables are to be stored as Parquet format

    """

    # get filepath to log data file
    log_path = Path(input_data) / Path("log-data/*.json")
    log_data = glob.glob(log_path, recursive=True)

    # log data schema
    log_schema = StructType([
        StructField("artist", StringType()),
        StructField("auth", StringType()),
        StructField("firstName", StringType()),
        StructField("gender", StringType()),
        StructField("itemInSession", IntegerType()),
        StructField("lastName", StringType()),
        StructField("length", DoubleType()),
        StructField("level", StringType()),
        StructField("location", StringType()),
        StructField("method", StringType()),
        StructField("page", StringType()),
        StructField("registration", DoubleType()),
        StructField("sessionId", IntegerType()),
        StructField("song", StringType()),
        StructField("status", IntegerType()),
        StructField("ts", LongType()),
        StructField("userAgent", StringType()),
        StructField("userId", StringType())
    ])
    # read log data file
    df_log = spark.read.json(log_data, log_schema)

    # filter by actions for song plays
    df_log = df_log.filter(df_log.page == "NextSong")

    # columns for user table
    users_cols_log = ["userId as user_id",
                      "firstName as first_name",
                      "lastName as last_name",
                      "gender",
                      "level"]
    # users table
    users_table = extract_cols(col_names=users_cols_log,
                               drop_dup_cols=["user_id"],
                               df=df_log,
                               partitionBy=None,
                               table_name='users',
                               if_use_select_expr=True)

    # create timestamp column from original timestamp column
    get_timestamp = udf(
        lambda x: datetime.fromtimestamp(
            (x / 1000.0)),
        T.TimestampType())
    # adding timestamp column to df_log
    df_log = df_log.withColumn("timestamp", get_timestamp(df_log.ts))

    # columns for time table
    time_cols_log = ["timestamp as start_time",
                     "HOUR(timestamp) as hour",
                     "DAY(timestamp) as day",
                     "WEEKofYEAR(timestamp) as week",
                     "MONTH(timestamp) as month",
                     "YEAR(timestamp) as year",
                     "DAYofWEEK(timestamp) as weekday"]

    # time table
    time_table = extract_cols(col_names=time_cols_log,
                              drop_dup_cols=["start_time"],
                              df=df_log,
                              partitionBy=["year", "month"],
                              table_name='time_table',
                              if_use_select_expr=True)

    # read in song data to use for songplays table
    songPath = Path(input_data) / Path("song-data/**/*.json")
    song_data = glob.glob(songPath, recursive=True)
    # schema for song dataset
    song_schema = StructType([
        StructField("artist_id", StringType()),
        StructField("artist_latitude", DoubleType()),
        StructField("artist_location", StringType()),
        StructField("artist_longitude", DoubleType()),
        StructField("artist_name", StringType()),
        StructField("duration", DoubleType()),
        StructField("num_songs", IntegerType()),
        StructField("song_id", StringType()),
        StructField("title", StringType()),
        StructField("year", IntegerType())
    ])
    song_df = spark.read.json(song_data, song_schema)
    # create temporary views to use them in the join
    song_df.createOrReplaceTempView("songs")
    df_log.createOrReplaceTempView("logs")

    # extract columns from joined song and log datasets to create songplays
    # table
    songplays_table = spark.sql("""
                                        SELECT l.timestamp AS start_time,
                                               l.userId AS user_id,
                                               l.level AS level,
                                               s.song_id AS song_id,
                                               s.artist_id AS artist_id,
                                               l.sessionId AS session_id,
                                               l.location AS locatioin,
                                               l.userAgent AS user_agent,
                                               YEAR(l.timestamp) as year,
                                               MONTH(l.timestamp) as month
                                        FROM logs l
                                        JOIN songs s ON (l.song=s.title
                                                         AND l.length=s.duration
                                                         AND l.artist=s.artist_name)
                                    """).withColumn("songplay_id", F.monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(
        Path(output_data) /
        Path("songplays.parquet"),
        partitionBy=["year", "month"],
        mode="overwrite")


def main():
    global output_data
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = ""

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)
    spark.stop()


if __name__ == "__main__":
    main()
