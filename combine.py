import pyspark as py
import errors
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType
from pyspark.sql.functions import unix_timestamp, from_unixtime
from pyspark.sql.functions import datediff, to_date, lit

class Combine:
    def __init__(self, dataframe, datediff=10):
        self.datediff= self.check_input(self, datediff)
        self.datediff = datediff
        self.dataframe = dataframe
        self.dataframe.show()
        self.split()
        self.combine()
    
    def self.check_input(self, datediff):
        if not isinstance(datediff, int) or datediff < 0:
            raise errors.InputError("The date differences cannot be smaller than 0")
        return datediff
    
    def split(self):
        self.comments = self.get_comments()
        self.health_events = self.get_health_events()
        
    def remove_null(self):
        self.dataframe = self.dataframe.where(self.AnimalId != "None")
        
    def get_comments(self):
        comments = self.dataframe.where((self.dataframe.Event == "Comment"))
        comments = comments.sort(comments["EventDate"])
        comments = comments.withColumn("Date", to_date(col("EventDate")))
        return comments.select(
            "AnimalId",
            'Date', 
            (col("Metadata").Message).alias('Comment')
        )
    
    def get_health_events(self):
        health = self.dataframe.where((self.dataframe.Event == "Health"))
        health = health.sort(health["EventDate"])
        health = health.withColumn("Date", to_date(col("EventDate")))
        return health.select(
            "AnimalId",
            'Date', 
            (col("Metadata").HealthCondition).alias('HealthCondition')
        )
        
    def combine(self):
        self.combined = self.health_events.join(self.comments, [(self.health_events.AnimalId == self.comments.AnimalId), (datediff(self.health_events.Date, self.comments.Date) <= self.datediff), (datediff(self.health_events.Date, self.comments.Date) >= -self.datediff)], "inner")
        
    def get_dataframe(self):
        return self.combined
                        