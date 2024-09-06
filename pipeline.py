from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import regexp_replace, col, to_date
from pyspark.sql import DataFrame
import psycopg2
from psycopg2 import sql
import logging
import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_soup(url: str) -> None:
    """fetching and parsing the html content of the given url"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return bs(response.text, 'html.parser')
    except requests.RequestException as e:
        logger.error(f"error fetching the url:{e}")
        return None

def get_unique_csv_links(url: str, data_format="csv") -> set:
    """extracting unique csv links from the html page"""
    unique_links = set()
    soup = get_soup(url)
    if soup:
        for link in soup.find_all('a'):
            file_link = link.get('href')
            if file_link and data_format in file_link:
                unique_links.add(file_link)
    return unique_links

def load_csv_to_pandas(csv_url: str) -> pd.DataFrame:
    """loading csv data into a pandas df"""
    columns = [
        'Serial Number', 'List Year', 'Date Recorded', 'Town', 'Address',
        'Assessed Value', 'Sale Amount', 'Sales Ratio', 'Property Type',
        'Residential Type', 'Location'
    ]
    try:
        df = pd.read_csv(csv_url, usecols=columns)
        return df
    except Exception as e:
        logger.error(f"error loading csv file to pandas:{e}")
        return pd.DataFrame()

def load_csv_to_spark(pandas_df: pd.DataFrame) -> DataFrame:
    """converting pandas df to Spark df"""
    spark = SparkSession.builder.appName("pandas df to Spark df").getOrCreate()

    schema = StructType([
        StructField('Serial Number', StringType(), True),
        StructField('List Year', IntegerType(), True),
        StructField('Date Recorded', StringType(), True),
        StructField('Town', StringType(), True),
        StructField('Address', StringType(), True),
        StructField('Assessed Value', FloatType(), True),
        StructField('Sale Amount', FloatType(), True),
        StructField('Sales Ratio', FloatType(), True),
        StructField('Property Type', StringType(), True),
        StructField('Residential Type', StringType(), True),
        StructField('Location', StringType(), True)
    ])

    spark_df = spark.createDataFrame(pandas_df, schema=schema)

    renamed_df = spark_df \
        .withColumnRenamed('Serial Number', 'SN') \
        .withColumnRenamed('List Year', 'list_year') \
        .withColumnRenamed('Date Recorded', 'date_recorded') \
        .withColumnRenamed('Town', 'town') \
        .withColumnRenamed('Address', 'address') \
        .withColumnRenamed('Assessed Value', 'assessed_value') \
        .withColumnRenamed('Sale Amount', 'sale_amount') \
        .withColumnRenamed('Sales Ratio', 'sales_ratio') \
        .withColumnRenamed('Property Type', 'property_type') \
        .withColumnRenamed('Residential Type', 'residential_type') \
        .withColumnRenamed('Location', 'location')
    renamed_df = renamed_df.repartition(100)
    return renamed_df

def clean_data(spark_df: DataFrame) -> DataFrame:
    """cleaning the spark df"""

    spark_df = spark_df.fillna('')
    spark_df = spark_df.withColumn("sales_ratio", regexp_replace(col("sales_ratio"), '[^\d.]', '').cast(FloatType())).filter(col("sales_ratio") >= 0)
    spark_df = spark_df.withColumn("sale_amount", regexp_replace(col("sale_amount"), '[^\d.]', '').cast(FloatType())).filter(col("sale_amount") >= 0)
    spark_df = spark_df.withColumn("date_recorded", to_date(col("date_recorded"), "MM/dd/yyyy"))
    spark_df = spark_df.withColumn("list_year", col("list_year").cast(IntegerType()))
    spark_df = spark_df.dropDuplicates(["SN"])
    return spark_df

def create_table_if_not_exists(connection, schema_name) -> None:
    """creating table if not exist"""
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL('''
                CREATE TABLE IF NOT EXISTS {}.real_estate_sales 
                (
                    SN VARCHAR(255),
                    list_year INTEGER,
                    date_recorded DATE,
                    town VARCHAR(255),
                    address VARCHAR(255),
                    assessed_value FLOAT,
                    sale_amount FLOAT,
                    sales_ratio FLOAT,
                    property_type VARCHAR(255),
                    residential_type VARCHAR(255),
                    location VARCHAR(255)
                )''').format(sql.Identifier(schema_name)))
            logger.info("table created successfully")
    except Exception as e:
        logger.error(f"failed table creation:{e}")

def insert_to_postgres(spark_df: DataFrame, connection, config) -> None:
    """inserting data from Spark df to postgres table"""
    try:
        spark_df.write \
            .format('jdbc') \
            .mode('append') \
            .option('url', f'jdbc:postgresql://{config.host}:{config.port}/{config.db}') \
            .option('dbtable', f'{config.schema_name}.real_estate_sales') \
            .option('user', config.user) \
            .option('password', config.password) \
            .save()
        logger.info("data inserted successfully")
    except Exception as e:
        logger.error(f"failed data insertion: {e}")

def main() -> None:
    """starting the ETL process from here"""
    # Database connection
    try:
        connection = psycopg2.connect(
            host=config.host,
            user=config.user,
            password=config.password,
            database=config.db,
            port=config.port
        )
        logger.info("database connected successfully")
    except Exception as e:
        logger.error(f"failed database connection: {e}")
        return

    # Step 1: Scraping the CSV file from the source
    unique_links = get_unique_csv_links(config.url)
    if not unique_links:
        logger.info("no any csv links found")
        return

    logger.info("unique csv links are")
    for link in unique_links:
        logger.info(link)

    # taking the first csv link
    csv_url = list(unique_links)[0]
    # Step 2:loading csv file into pandas df
    pandas_df = load_csv_to_pandas(csv_url)
    if pandas_df.empty:
        logger.info("no any data loaded from csv file")
        return
    # Step 3:loading pandas df into Spark df
    spark_df = load_csv_to_spark(pandas_df)
    # Step 4:cleaning data
    spark_df = clean_data(spark_df)
    logger.info("Finished cleaning Spark DataFrame")
    spark_df.show()
    # Step 5:creating table if not exists
    create_table_if_not_exists(connection, config.schema_name)
    # Step 6:inserting data into postgres
    insert_to_postgres(spark_df, connection, config)

    spark_df.sparkSession.stop()

    # closing db connection
    connection.close()

if __name__ == "__main__":
    main()
