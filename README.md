# classify_cattle_disease
Code to preprocess and classify the dataset


# Combine health events and comments

To combine health events and comments you can use the 'combine' class. It takes the 'AllEventsMapped' as its dataframes and a datedifference to match. The datedifference is set to 10 on default. Note that one has too go through the results manually to remove any noise. It is not possible to match one on one since that results in almost no results.

# Translate

To translate the comments of a dataframe to English, the 'translate' class can be used. The translate class will detect the language, correct the spelling, and translate the comments that are non-English. Out of the box, it will use the stochastic gradient descent algorithm of SVM with 600 features and bigrams. The possible algorithms are:
- Naive Bayes
- Bernoulli Bayes
- Multinomial Bayes
- Logistic Regression
- SGD (Stochastic Gradient Descent version of SVM)

All algorithms support the following n-gram feature combinations:
- 1: 300
- 2: 300, 600, 1200, 2000
- 3: 300, 600, 1200, 3000
- 4: 300, 600, 1200, 2000
- 5: 300, 600, 1200

The class uses google translate to translate the comments to English, although this part is currently not working.

# Preprocessing

The 'preprocess' class takes care of stop word removal, enrichment, stemming, and lemmitization. The comments are enriched based on a lookup dictionary containing information about medicines and diseases. Of the comment contains neither of those, it is enriched using the TAGME procedure based on Wikipedia articles. The named entities are recognized using the Stanford NER tagger. The resulting dataframe will contain enriched, stemmed comments.

# Classification
The 'classify' class takes care of the classifying and prediction of comments based on the health conditions. It currently supports the LDA classification, L-LDA classification and the more classical approaches.

## Latent Dirichlet Allocation
The LDA classifier takes in a combination of topics, passes, decay rate, iterations, and minimum probability. It uses a lookup dictionary to interpret the results. The models are stored in 'models/unsupervised'.
The possible combinations are:
- 6 topics, 100 passes, decay rate 6, 10 iterations
- 6 topics, 250 passes, decay rate 6, 10 iterations
- 6 topics, 500 passes, decay rate 6, 10 iterations
- 7 topics, 250 passes, decay rate 6, 10 iterations
- 8 topics, 100 passes, decay rate 6, 10 iterations
- 8 topics, 250 passes, decay rate 6, 10 iterations 
- 8 topics, 500 passes, decay rate 6, 10 iterations
- 9 topics, 250 passes, decay rate 6
- 11 topics, 100 passes, decay rate 6, 10 iterations
- 11 topics, 250 passes, decay rate 6, 10 iterations
- 11 topics, 500 passes, decay rate 6, 10 iterations
- 13 topics, 100 passes, decay rate 6, 10 iterations
- 13 topics, 250 passes, decay rate 6, 10 iterations
- 13 topics, 500 passes, decay rate 6, 10 iterations

The minimum probability can be set as a value between 0 and 1, it is the threshold on which the algorithm will return the result. If the algorithm returns a possibility score higher than the threshold, the comment is classified accordingly.
The lookups can be found in lookups.txt

## Labeled Latent Dirichlet Allocation

The L-LDA classifier takes a combintion of topics, passes, and a minimum probability. 

The possible combinations are:
- 6 topics, 50 passes
- 6 topics, 150 passes
- 6 topics, 250 passes
- 7 topics, 50 passes
- 7 topics, 150 passes
- 7 topics, 250 passes
- 8 topics, 50 passes
- 8 topics, 150 passes
- 8 topics, 250 passes
- 10 topics, 50 passes
- 10 topics, 150 passes
- 10 topics, 250 passes
- 13 topics, 50 passes
- 13 topics, 150 passes
- 13 topics, 250 passes

The minimum probability can be set as a value between 0 and 1, it is the threshold on which the algorithm will return the result. If the algorithm returns a possibility score higher than the threshold, the comment is classified accordingly.
The lookups can be found in lookups.txt

## Classical approaches
The algorithms used in the classical approach are: 1) Naive Bayes, 2) Logistic Regression, 3) Multinomial Bayes, 3) SGD (version of SVM). It takes a number of topics, n-grams, features, and lookup
The possible topics are: 1) 7, 2) 8, 3) 11, 4) 13.

The possible combinations of n-grams and features are:
- 1 ngram: 1000 features, 2500 features, 5000 features.
- 2 ngram: 3000 features, 10000 features, 20000 features.
- 3 ngram: 3000 features, 10000 features, 20000 features.



