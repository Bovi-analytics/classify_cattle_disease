import pyspark as py
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
import re
import codecs
import numpy as np
import string
import pickle
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from googletrans import Translator
import spellcheck


    
class Translate:
    
    switcher = {
        "naive": "",
        "bernoulli": "BernoulliNB_",
        "lr": "LogisticRegression_",
        "multinomial": "MNB_",
        "SGD": "SGDClassifier_"
    }
    
    translator = Translator()
    
    translate_switcher = {
        "Dutch": "nl",
        "English": "en",
        "French": "fr"
    }
    
    corrector = {
        'en': spellcheck.EnglishSpellCheck(),
        'nl': spellcheck.DutchSpellCheck(),
        'fr': spellcheck.FrenchSpellCheck()
    }
    
    gram_feature_combinations = {
        1: [300],
        2: [300, 600, 1200, 2000],
        3: [300, 600, 1200, 3000],
        4: [300, 600, 1200, 2000],
        5: [300, 600, 1200]
    }
        
    def __init__(self, dataframe, n=2, features = 600, algorithm="SGD", target="en", text_column = "Comments"):
        self.n = n
        self.features = features 
        self.algorithm = "naive"
        self.target = "en"
        self.text_column = "Comments"
        if not self.check_feature_ngram():
            print("Wrong parameter settings")
            return
        self.load_detect_algorithm()
        dataframe = self.detect_languages(dataframe)
        dataframe = self.correct_spelling(dataframe)
        #dataframe = self.translate(dataframe)
        self.dataframe = dataframe.select(
            (col("corrected")).alias(self.text_column)
        )
        self.dataframe.show()
    
    def get_dataframe(self):
        return self.dataframe
    
    def translate(self, dataframe):
        translateFunc = F.udf(self.get_translation, StringType())
        dataframe = dataframe.withColumn("translated", translateFunc("corrected", "language"))
        return dataframe
    
    def correct_spelling(self, dataframe):
        correctFunc = F.udf(self.correct_comment, StringType())
        dataframe = dataframe.withColumn("corrected", correctFunc(self.text_column, "language"))
        return dataframe
    
    def get_languages(self, dataframe):
        detectFunc = F.udf(self.detect_language, StringType())
        dataframe = dataframe.withColumn("language", detectFunc(self.text_column))
        return dataframe
        
    def detect_languages(self, dataframe):
        detectFunc = F.udf(self.detect_language, StringType())
        dataframe = dataframe.withColumn("language", detectFunc(self.text_column))
        return dataframe
    
    def check_feature_ngram(self):
        if self.features in self.gram_feature_combinations[self.n]:
            return True
        return False
    
    def load_detect_algorithm(self):
        try:
            f = open('language detection/' + str(self.n) + '-ngram/n-'+ str(self.features)+'-featuresets.pickle', 'rb')
            self.featureset = pickle.load(f)
            f.close()
            f = open('language detection/' + str(self.n)+'-ngram/n-'+str(self.features)+'-'+self.switcher[self.algorithm]+"classifier.pickle", 'rb')
            self.language_detection_algorithm = pickle.load(f)
            f.close()
        except:
            print("Could not load models")
    
    def detect_language(self, line):
        line = self.preprocess(line)
        ngrams = self.get_ngrams(line)
        features = self.get_features(ngrams)
        return self.language_detection_algorithm.classify(features)
    
    def get_features(self, grams):
        to_return = {}
        if isinstance(grams, list):
            for gram in grams:
                found = False
                for sen in self.featureset:
                    if gram == sen:
                        found = True
                        to_return[gram] = True
                if not found:
                    to_return[gram] = False
        return to_return
    
    def preprocess(self, line):
        if line != "" and line is not None:
            line = " ".join(line.split()[0:])
            line = line.lower()
            line = re.sub(r"\d+", "", line)
            line = line.translate(str.maketrans('', '', string.punctuation))
        return line

    def get_ngrams(self, line):
        detected_ngrams = nltk.ngrams(line, self.n)
        return detected_ngrams

    def create_ngram_features(self, line):
        ngrams = dict()
        sequence = preprocess(line)
        detected_ngrams = self.get_ngrams(sequence, self.n)
        for detected in detected_ngrams:
            ngrams[detected] = ngrams.get(detected, 0) + 1
        return sorted(ngrams.items(), key=lambda item: item[1],reverse=True)
    
    def correct_comment(self, comment, language):
        return self.corrector[self.translate_switcher[language]].correct_sentence(comment)
    
    def get_translation(self, comment, language):
        if self.translate_switcher[language] == "en":
            return comment
        
        return self.translator.translate(comment, src=self.translate_switcher[language], dest="en").text
    