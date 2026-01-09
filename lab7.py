#1. Using Naive Bayes classifier described in this chapter, and any features you can think of, build the best name gender classifier you can. Begin by splitting the Names Corpus into three subsets: 500 words for the test set, 500 words for the dev-test set, and the remaining 6,900 words for the training set. Then, starting with the example name gender classifier, make incremental improvements. Use the devtest set to check your progress. Once you are satisfied with your classifier, check its final performance on the test set. How does the performance on the test set compare to the performance on the dev-test set? Is this what you’d expect?
import random
import nltk
from nltk import accuracy, WordNetLemmatizer
from nltk.corpus import names, movie_reviews
from nltk.corpus import senseval
from nltk.corpus.reader import documents

# # nltk.download('names')
#
# labeled_names = ([(name ,'male') for name in names.words('male.txt')] +  [(name,"female") for name in names.words('female.txt')])
#
# random.shuffle(labeled_names)
#
# test_names = labeled_names[:500]
# devtest_names = labeled_names[500:1000]
# train_names = labeled_names[1000:]
#
# def gender_features(name):
#     return {'last letter:' :name[-1].lower()}
#
# train_set = [(gender_features(n), g) for (n, g) in train_names]
# devtest_set = [(gender_features(n), g) for (n, g) in devtest_names]
# test_set = [(gender_features(n), g) for (n, g) in test_names]
#
# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(classifier.classify(gender_features('Neo')))
#
# print("accuracy:", nltk.classify.accuracy(classifier,devtest_set))

#2. The Senseval 2 Corpus contains data intended to train word-sense disambiguation classifiers.
# It contains data for four words: hard, interest, line, and serve. Choose one of these four words, and load the corresponding data:

# instances = senseval.instances('hard.pos')
#
# size = int(len(instances)*0.1)
# train_set ,test_set = instances[size:],instances[:size]
# print(f"Total instances: {len(instances)}")
# print(f"Train: {len(train_set)} | Test: {len(test_set)}")
#
# def sense_features(instance):
#     features ={}
#     position= instance.position
#
#     context = []
#     for token in instance.context:
#         if isinstance(token,tuple): context.append(token[0].lower())
#         else: context.append(str(token).lower())
#
#     if position > 0:
#         features['prev_word'] = context[position - 1]
#     if position < len(context) - 1:
#         features['next_word'] = context[position + 1]
#     for word in context:
#         features[f'contains({word})'] = True
#     return features
#
# train_features = [(sense_features(i), i.senses[0]) for i in train_set]
# test_features = [(sense_features(i),i.senses[0]) for i in test_set]
#
# classifier = nltk.NaiveBayesClassifier.train(train_features)
# print("accuracy:" , nltk.classify.accuracy(classifier,test_features))
# classifier.show_most_informative_features(10)


#3. Using the movie review document classifier, generate a list of the 30 features that the classifier finds to be most informative. Can you explain why these particular features are informative? Do you find any of them surprising?

# nltk.download('movie_reviews')
#
# documents = [list((movie_reviews.words(fileid),category)) for category in movie_reviews.categories() for fileid in movie_reviews.fileids()]
# random.shuffle(documents)
#
# all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
#
# word_features= list(all_words.keys())[:2000]
# print(word_features)
# def document_features(document):
#     document_words = set(w.lower() for w in document)
#     features = {}
#     for word in word_features:
#         features[f'contains({word})'] = (word in document_words)
#     return features
# featuresets = [(document_features(d), c) for (d, c) in documents]
#
# train_set = featuresets[100:]
# test_set = featuresets[:100]
#
# classifier = nltk.NaiveBayesClassifier.train(train_set)
#
# accuracy = nltk.classify.accuracy(classifier,test_set)
# print(accuracy)
# classifier.show_most_informative_features(30)

#4. Select one of the classification tasks described, such as name gender detection, document classification or part-of-speech tagging. Using the same training and test data, and the same feature extractor, build two classifiers for the task: a decision tree and a naive Bayes classifier. Compare the performance of the two classifiers on your selected task. How do you think that your results might be different if you used a different feature extractor?
#
# labeled_names = ([(name,"female") for name in names.words('female.txt')] + [(name,"male") for name in names.words('male.txt')])
# random.shuffle(labeled_names)
#
# def gender_features(name):
#     name = name.lower()
#     return {
#         'last_letter': name[-1],
#         'last_two': name[-2:]
#     }
#
# split_point = int(len(labeled_names) * 0.8)
# train_names = labeled_names[:split_point]
# test_names = labeled_names[split_point:]
#
# train_set = [(gender_features(n), g) for (n, g) in train_names]
# test_set = [(gender_features(n), g) for (n, g) in test_names]
#
# nb_classifier = nltk.NaiveBayesClassifier.train(train_set)
# nb_acc = nltk.classify.accuracy(nb_classifier,test_set)
# print(nb_acc)
# dt_classifier = nltk.DecisionTreeClassifier.train(train_set)
# dt_acc = nltk.classify.accuracy(dt_classifier,test_set)
# print(dt_acc)
#5. Word features can be very useful for performing document classification, since the words that appear in a document give a strong indication about what its semantic content is.
#However, many words occur very infrequently, and some of the most informative words in a document may never have occurred in our training data.
#One solution is to make use of a lexicon, which describes how different words relate to one another. Using the WordNet lexicon, augment the movie review document classifier presented in this chapter to use features that generalize the words that appear in a document, making it more likely that they will match words found in the training data.
import nltk
from nltk.corpus import movie_reviews, wordnet
from nltk.corpus import stopwords
import random

from pyexpat import features
from unicodedata import category

# nltk.download('movie_reviews')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4')


documents = [(list(movie_reviews.words(fileid)),category)  for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

lemmatizer = WordNetLemmatizer()

def get_wordnet_features(word):
    features = set()
    word = word.lower()
    lemma = lemmatizer.lemmatize(word)
    features.add(lemma)

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            features.add(l.name().lower())

        for hyper in syn.hypernyms():
            for l in hyper.lemmas():
                features.add(l.name().lower())
    return features
print(get_wordnet_features("wonderful"))

stop_words = set(stopwords.words("english"))

def document_features(document):
    document_words = set(w.lower() for w in document if w.isalpha() and w.lower() not in stop_words)
    features = {}

    for word in document_words:
        features[f'contains({word})'] = True

        for related_word in get_wordnet_features(word):
            features[f'contains_lexical({related_word})'] = True
    return features

featuressets = [(document_features(d),c) for (d,c) in documents]

train_set , test_set = featuressets[100:],featuressets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Accuracy:", nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(15)