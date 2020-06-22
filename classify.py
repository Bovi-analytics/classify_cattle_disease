import os 
import pyspark as py
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import *
import gensim
from gensim.test.utils import datapath
import errors
from importlib import reload
from nltk import word_tokenize
import nltk
import pickle
import labeled_lda as llda
reload(errors)

class Classification:
    
    def __init__(self):
        print("test")
        
    def check_input(self):
        pass
    
    def load_model(self):
        pass
    
    def classify(self):
        pass
    
    def get_classification(self):
        pass
        
        
class UnsupervisedClassification(Classification):
    base_path = os.getcwd() + "\\models\\unsupervised\\"
    
    possible_topics = set([7,8,9,11,13])
    
    possible_passes = set([100, 250, 500])
    
    possible_decay = set([6])
    
    possible_iterations = set([10])
    
    def __init__(self, dataframe, lookup, topics=7, passes=250, decay=6, iterations=10, min_probability=0.2, text_column="Comments"):
        dataframe, self.lookup, self.topics, self.passes, self.decay, self.iterations, self.min_probability, self.text_column = self.check_input(dataframe, lookup, topics, passes, decay, iterations, min_probability, text_column)
        self.load_model()
        dataframe = self.classify(dataframe)
        dataframe.show()
    
    def get_classification(self, comment):
        to_return = []
        comment = word_tokenize(comment)
        comment = self.dictionary.doc2bow(comment)
        predictions = self.model[comment]
        for prediction in predictions:
            if prediction[1] > self.min_probability:
                to_return.extend(self.lookup[prediction[0]])
        return str(to_return)
    
    def classify(self, dataframe):
        classifyFunc = F.udf(self.get_classification, StringType())
        dataframe = dataframe.withColumn("classification", classifyFunc(self.text_column))
        return dataframe
        
    def check_input(self, dataframe, lookup, topics, passes, decay, iterations, min_probability, text_column):
        if dataframe == None or lookup == None or topics == None or passes == None or min_probability == None:
            raise errors.InputError("Please provide all necessary information")
        if topics not in self.possible_topics:
            raise errors.InputError("We do not have a topic with that number of topics")
        if passes not in self.possible_passes:
            raise errors.InputError("We do not have a topic with that number of passes")
        if decay is not None and decay not in self.possible_decay:
            raise errors.InputError("We do not have a topic with that decay rate")
        if iterations is not None and iterations not in self.possible_iterations:
            raise errors.InputError("We do not have a topic with that iteration rate")
        if not isinstance(text_column, str) or text_column not in dataframe.columns:
            raise errors.InputError("That column does not exist")
        return(dataframe, lookup, str(topics), str(passes), str(decay), str(iterations), min_probability, str(text_column))
    
    def build_path(self):
        path = self.base_path + "lda-category-n"+self.topics+"w7"+"p"+self.passes
        if self.decay != "" and self.decay != "None":
            path +="d"+self.decay
        if self.iterations != "" and self.iterations != "None":
            path +="i"+self.iterations
        return path       
    
    def load_model(self):
        location = datapath(self.build_path())
        self.model = gensim.models.LdaMulticore.load(location)
        with open(self.base_path + 'document_set.pickle', 'rb') as f:
            self.dictionary = pickle.load(f)
            
            
class SupervisedClassification(Classification):
    base_path = os.getcwd() + "\\models\\supervised\\llda-"
    
    def __init__(self, dataframe, lookup, topics=7, passes=250, min_probability=0.2, text_column="Comments"):
        self.text_column = text_column
        dataframe, self.lookup, self.topics, self.passes, self.min_probability, self.text_column = self.check_input(dataframe, lookup, topics, passes, min_probability, text_column)
        self.load_models()
        dataframe = self.classify(dataframe)
        dataframe.show()
    
    def get_classification(self, comment):
        to_return = []
        predictions =  self.model.inference(comment)
        for prediction in predictions:
            if prediction[1] > self.min_probability:
                to_return.extend(self.lookup[prediction[0]])
        return str(to_return)
    
    def classify(self, dataframe):
        classifyFunc = F.udf(self.get_classification, StringType())
        dataframe = dataframe.withColumn("classification", classifyFunc(self.text_column))
        return dataframe
        
    def build_path(self):
        return self.base_path + self.topics + "category-t"+self.passes
    
    def load_models(self):
        self.model = llda.LldaModel()
        self.model.load_model_from_dir(self.build_path(), load_derivative_properties=False)
        
    def check_input(self, dataframe, lookup, topics, passes, min_probability, text_column):
        if dataframe == None or lookup == None or topics == None or passes == None or min_probability == None:
            raise errors.InputError("Please provide all necessary information")
        if not isinstance(text_column, str) or text_column not in dataframe.columns:
            raise errors.InputError("That column does not exist")
        return(dataframe, lookup, str(topics), str(passes), min_probability, str(text_column))
    
class ClassicalClassification(Classification):
    base_path = os.getcwd() + "\\models\\classical\\"
    
    classifier_type_lookup = {
        "sgd": "SGDClassifier",
        "lr": "LogisticRegression",
        "mb": "multinb",
        "nb": "naive-bayes"
    }
    
    def __init__(self, dataframe, lookup, classifier_type="nb", topics=7, n_grams=3, features=10000, text_column="Comments"):
        self.text_column = text_column
        dataframe, self.lookup, self.topics, self.classifier_type, self.n_grams, self.features, self.text_column = self.check_input(dataframe, lookup, classifier_type, topics, n_grams, features, text_column)
        self.load_models()
        dataframe = self.classify(dataframe)
        dataframe.show()
    
    def get_ngrams(self, comment):
        if int(self.n_grams) == 1:
            return nltk.word_tokenize(comment)
        if int(self.n_grams) == 2:
            return list(nltk.bigrams(nltk.word_tokenize(comment)))
        if int(self.n_grams) == 3:
            return list(nltk.trigrams(nltk.word_tokenize(comment)))
        
    def get_classification(self, comment):
        ngrams = self.get_ngrams(comment)
        features = self.get_features(ngrams)
        return self.lookup[self.model.classify(features)]
        
    def get_features(self, grams):
        to_return = {}
        for gram in grams:
            found = False
            for sen, category in self.dictionary:
                for iets in sen:
                    if gram == iets:
                        found = True
                        to_return[gram] = True
            if not found:
                to_return[gram] = False
        return to_return
    
    def classify(self, dataframe):
        classifyFunc = F.udf(self.get_classification, StringType())
        dataframe = dataframe.withColumn("classification", classifyFunc(self.text_column))
        return dataframe
    
    def build_path(self):
        return self.base_path + self.topics + "cat\\" + self.n_grams + "-ngram\\n" + self.features + "\\"
    
    def load_models(self):
        base_path = self.build_path()
        with open(base_path + 'documents.pickle', 'rb') as f:
            self.dictionary = pickle.load(f)
        with open(base_path + self.classifier_type_lookup[self.classifier_type] + ".pickle", "rb") as f:
            self.model = pickle.load(f)
    
    def check_input(self, dataframe, lookup, classifier_type, topics, n_grams, features, text_column):
        if dataframe == None or lookup == None or classifier_type == None or topics == None or n_grams == None or features == None:
            raise errors.InputError("Please provide all necessary information")
        if not isinstance(text_column, str) or text_column not in dataframe.columns:
            raise errors.InputError("That column does not exist")
        return (dataframe, lookup, str(topics), classifier_type, str(n_grams), str(features), str(text_column))
    