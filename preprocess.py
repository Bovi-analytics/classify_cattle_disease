# This class will detect the different text snippet types and preprocess them based on that type, the different types are:
# - non cow related (no AnimalId) -> removed
# - nonsensical comments -> removed
# - containing disease/medicine information -> the dosage is removed, enriched on disease/medicine information 
# - non-specific comments -> cleaned using normal rules, enriched using tagme


import pyspark as py
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
from contractions import expandContractions
import tagme
import errors
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer

class Preprocess:
    
    tagme_token = "967d886e-5b07-48d8-91fd-9fd66c8ec034-843339462"
    
    abbreviation_files = {
        "dutch": "abbreviations\\afkortingen-nl.txt",
        "english": "abbreviations\\afkortingen-en.txt",
        "french": "abbreviations\\afkortingen-fr.txt"
    }
    
    abbreviations = {}
    
    def __init__(self, dataframe, text_column="Comments"):
        self.text_column = self.check_text_column(dataframe, text_column)
        self.set_ner_tagger()
        self.set_stopwords()
        self.load_medicine_lookup()
        self.load_condition_lookup()
        self.set_english_words()
        self.load_abbreviation_lookup()
        self.set_stemmer()
        self.set_condition_specific_tags()
        dataframe = self.general_cleanup(dataframe)
        dataframe = self.type_specific_cleanup(dataframe)
        dataframe = self.enrich(dataframe)
        dataframe = self.stem_text(dataframe)
        self.dataframe = dataframe 
    
    def load_abbreviation_lookup(self):
        for language in self.abbreviation_files.keys():
            self.abbreviations[language] = {} 
            with open(self.abbreviation_files[language], "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.split(":")
                    line[1] = line[1].replace("\n", "")
                    self.abbreviations[language][line[0]] = line[1]
    
    def check_text_column(self, dataframe, text_column):
        if not isinstance(text_column, str) or text_column not in dataframe.columns:
            raise errors.InputError("That column does not exist")
        return text_column
    
    def get_dataframe(self):
        return self.dataframe
    
    def set_stemmer(self):
        self.stemmer = SnowballStemmer("english")
    
    def set_ner_tagger(self):
        self.st = StanfordNERTagger('stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')
        
    def set_stopwords(self):
        self.general_stop_words = set(stopwords.words('english'))
        
    def load_medicine_lookup(self):
        with open("enrich_info/medicine.pickle", 'rb') as f:
            self.medicine_dict = pickle.load(f)
        
    def load_condition_lookup(self):
        with open("enrich_info/condition.pickle", 'rb') as f:
            self.condition_dict = pickle.load(f)
    
        
    def set_condition_specific_tags(self):
        self.specific_tags = ["liters", "liter", "cubic", "centimeter", "centimeters", "mililiters", "mililiter", "milk", "drip", "IV"]
    
    def set_english_words(self):
        self.words = set()
        with open("symspell/en_50k.txt") as f:
            for line in f:
                self.words.add(line.split()[0])
            
    def general_cleanup(self, dataframe):
        cleanFunc = F.udf(self.general_clean_comment, StringType())
        dataframe = dataframe.withColumn("cleaned", cleanFunc(self.text_column))
        return dataframe.select(
            (col("cleaned")).alias(self.text_column)
        )
    
    def get_stemming(self, comment):
        tokens = word_tokenize(comment)
        stemmed = []
        for token in tokens:
            stemmed.append(self.stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
        return ' '.join(stemmed)
    
    def stem_text(self, dataframe):
        stemmingFunc = F.udf(self.get_stemming, StringType())
        dataframe = dataframe.withColumn("stemmed", stemmingFunc(self.text_column))
        return dataframe.select(
            (col("stemmed")).alias(self.text_column)
        )
    
    def enrich(self, dataframe):
        annotateFunc = F.udf(self.get_enrichment, StringType())
        dataframe = dataframe.withColumn("annotated", annotateFunc(self.text_column))
        return dataframe.select(
            (col("annotated")).alias(self.text_column)
        )
    
    def type_specific_cleanup(self, dataframe):
        specificClean = F.udf(self.specific_clean_comment, StringType())
        dataframe = dataframe.withColumn("cleaned", specificClean(self.text_column))
        return dataframe.select(
            (col("cleaned")).alias(self.text_column)
        )
    
    def get_condition_enrichment(self, comment):
        tokens = word_tokenize(comment)
        for token in tokens:
            if token in self.medicine_dict.keys():
                comment += " " + self.medicine_dict[token]
            elif token in self.condition_dict.keys():
                comment += " " + self.condition_dict[token]
        return comment
    
    def get_normal_enrichment(self, comment):
        base_url = "https://en.wikipedia.org/wiki/"
        tagme.GCUBE_TOKEN = self.tagme_token
        annotations = tagme.annotate(comment)
        for annotation in annotations.get_annotations(0.4):
            response = requests.get(base_url + annotation.entity_title)
            soup = BeautifulSoup(response.text, 'html.parser')
            p = soup.find_all('p')
            wiki_text = ""
            for paragraph in p:
                wiki_text += paragraph.get_text() + " "
            comment += " " + wiki_text        
        return comment
    
    def get_type_for_enrichment(self, comment):
        if comment == "":
            return "empty"
        if self.contains_medicine(comment) or self.contains_condition(comment):
            return "condition"
        else:
            return "normal"
    
    def get_enrichment(self, comment):
        comment_type = self.get_type_for_enrichment(comment)
        if comment_type == "empty":
            return comment
        if comment_type == "condition":
            return self.get_condition_enrichment(comment)
        else:
            return self.get_normal_enrichment(comment)
    
    def specific_clean_comment(self, comment, animalId="set"):
        #not related to a cow
        if animalId == None:
            # emty strings will be removed in a later stage
            return ""
        #check whether it contains disease/medicine information
        if self.contains_medicine(comment) or self.contains_condition(comment):
            return self.clean_condition_comment(comment)
        #nonsense comment
        if self.check_nonsense_comment(comment):
            return ""
        else:
            return self.clean_normal_comment(comment)
        
    def check_nonsense_comment(self, comment):
        tokens = word_tokenize(comment)
        for token in tokens:
            if token != "score" and token != "and":
                if token in self.words:
                    return False
        return True
    
    def clean_condition_comment(self, comment):
        for tag in self.specific_tags:
            comment = comment.replace(tag, "")
        tokens = word_tokenize(comment)
        nertags = self.st.tag(tokens)
        for tag in nertags:
            if tag[1] != "O" and tag[0] not in self.medicine_dict.keys() and tag[0] not in self.condition_dict.keys():
                comment = comment.replace(tag[0], "")
        return ' '.join(comment.split())
    
    def clean_normal_comment(self, comment):
        tokens = word_tokenize(comment)
        nertags = self.st.tag(tokens)
        for tag in nertags:
            if tag[1] != "O":
                comment = comment.replace(tag[0], "")
        return ' '.join(comment.split())
    
    def contains_medicine(self, comment):
        medicine_keys = self.medicine_dict.keys()
        for key in medicine_keys:
            if key in comment:
                return True
        return False
    
    def contains_condition(self, comment):
        condition_keys = self.condition_dict.keys()
        for key in condition_keys:
            if key in comment:
                return True
            
        return False
    
    def split_integer_digit_string(self, comment):
        return ' '.join(re.split('(\d+)', comment))
    
    def remove_digits(self, comment):
        remove_digits = str.maketrans('', '', string.digits)
        return comment.translate(remove_digits)
    
    def remove_punctuation(self, comment):
        remove_punctuation = str.maketrans('', '', string.punctuation)
        return comment.translate(remove_punctuation)
    
    def general_clean_comment(self, comment):
        comment = comment.lower()
        comment = expandContractions(comment)
        comment = self.split_integer_digit_string(comment)
        comment = self.remove_digits(comment)
        comment = self.remove_punctuation(comment)
        tokenized = word_tokenize(comment)
        stop_word_removed = []
        for word in tokenized:
            if word not in self.general_stop_words:
                stop_word_removed.append(word)
        return ' '.join(stop_word_removed)
    