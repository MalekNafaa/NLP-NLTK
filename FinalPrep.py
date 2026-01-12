#THERE IS NO FINAL PREP BUT I WILL DO LAST YEAR EXAM I FOUND AS FINAL PREP
# بسم الله الرحمن الرحيم
import re

import matplotlib.pyplot as plt
import nltk
from nltk import FreqDist, ConditionalFreqDist, accuracy, word_tokenize, NaiveBayesClassifier, pos_tag
from nltk.corpus import gutenberg, brown, stopwords,wordnet as wn
from pyexpat import features
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# Write a Python program that performs the following steps on a given text file named task1 available here. Save it to the same directory as the script.
#1. (2pts) Read the content of the file.
#2. (6pts) Clean the text by
# Removing punctuation.
# Converting all words to lowercase.
    # Extracting any email addresses using regular expressions.
# def analyze_text_file(file_path):
#     with open(file_path,"r",encoding="utf-8") as file:
#         file_text = file.read()
#
#     text = []
#     pattern = "r[\w\s]"
#     clean_text = re.sub(pattern,"",file_text)
#     clean_text.lower()
#     email = re.findall(r"[a-zA-Z0-9.]+@[a-z]+\.[a-z]+",file_text)
#     # print(email)
#     # print(clean_text)
#     #3 (5pts) Tokenize the cleaned text into words.
#
#     tokenized_text = nltk.word_tokenize(clean_text)
#
#     #4 (5pts) Count the frequency of each word and display the top 10 most frequent words along with their counts.
#
#     freq_count = FreqDist(tokenized_text)
#     most_common = freq_count.most_common(10)
#
#     #5 (10 pts) Find and plot a conditional frequency distribution that shows how often each word length (in characters) appears for words starting with specific letters (e.g., 'a', 'b', 'c').
#
#     letters = ['a','b','c']
#
#     cfd = ConditionalFreqDist(
#         (word[0], len(word))
#         for word in tokenized_text
#         if word[0].lower() in letters
#     )
#     cfd.tabulate(conditions = letters)
#     # cfd.plot()
#     # plt.show()
#     return most_common,email
# print(analyze_text_file('task1.txt'))
#
# #TASK 2
# from nltk.corpus import wordnet as wn
# #1. (10 pts) Uses the Gutenberg corpus in NLTK to find the most common word and its frequency.
# #2. (10 pts) For the most common word, use WordNet to find its synonyms and antonyms.

# def msv():
#     all_words = gutenberg.words()
#     fd = FreqDist([w.lower() for w in all_words if w.isalpha() not in stopwords.words("english")])
#     most_common_word = fd.max()
#     frequency = fd.get(most_common_word)
#     #synonyms
#     synonyms = []
#     antonyms = []
#     for syn in wn.synsets(most_common_word):
#         synonyms+= syn.lemma_names()
#         for lemma in syn.lemmas():
#             antonyms += [l.name() for l in lemma.antoyms()]
#
#
#     """
#     Find the most common word in the Gutenberg corpus, and determine its
#     synonyms and antonyms using WordNet.
#     """
#     return  most_common_word,frequency, synonyms, antonyms
# # print(common_word_relationships())
# if __name__  == "__main__":
#     print(msv())
#Task 3
#(10 pts) Use the Brown corpus in NLTK to train a Bigram tagger using universal_tagset.
#(10 pts) Use this tagger to tag the parts of speech for the words in the following sentence:
#"The quick brown fox jumps over the lazy dog."
#(5 pts) Return the tagged sentence and the accuracy of the trained tagger on the Brown corpus.
# def categorize_and_tag():
#     """
#     Train a Bigram tagger, tag a given sentence, and evaluate accuracy.
#     """
#     tagged_sents = brown.tagged_sents(tagset="universal")
#     size = int(len(tagged_sents) * 0.1)
#     train_set = tagged_sents[size:]
#     test_set = tagged_sents[:size]
#     bigram_tagger = nltk.BigramTagger(train_set)
#
#     sentance = "The quick brown fox jumps over the lazy dog."
#     tokens = word_tokenize(sentance)
#     tagged_sentence = bigram_tagger.tag(tokens)
#     accuracy = bigram_tagger.accuracy(test_set)
#     # Your code here
#     return tagged_sentence, accuracy
# print(categorize_and_tag())

data = [
    {"review": "I loved the movie! The acting was great and the plot was engaging.", "sentiment": "positive"},
    {"review": "What a terrible movie. The story made no sense and the characters were dull.", "sentiment": "negative"},
    {"review": "Amazing cinematography and a gripping story. Highly recommend!", "sentiment": "positive"},
    {"review": "I didn't like it. The pacing was too slow.", "sentiment": "negative"},
    {"review": "A wonderful experience! Beautiful visuals and outstanding performances.", "sentiment": "positive"},
    {"review": "Horrible! This movie was a waste of time.", "sentiment": "negative"},
    {"review": "Good direction but the storyline could have been better.", "sentiment": "neutral"},
    {"review": "The soundtrack was fantastic, but the plot fell apart halfway.", "sentiment": "neutral"},
    {"review": "An excellent adaptation of the novel. Brilliant acting!", "sentiment": "positive"},
    {"review": "The movie was just okay. Nothing special about it.", "sentiment": "neutral"}
]
#TASK 1: Feature Extraction/Word Embeddings
#(10pts) Create a Bag of Words (BoW) or Term Frequency-Inverse Document Frequency (TF-IDF) matrix for the reviews.
#(5pts) Display the resulting feature matrix.

#BOW
# corpus = [d["review"] for d in data]

# bow_vectorizer = CountVectorizer()
# x = bow_vectorizer.fit_transform(corpus)
#
# print(bow_vectorizer.get_feature_names_out())
# print(x.toarray())

# TF
# tf_vectorizer = TfidfVectorizer()
# tf_x = tf_vectorizer.fit_transform(corpus)
# print(tf_vectorizer.get_feature_names_out())
# print(tf_x.toarray())

#TASK 2: Model Selection and Training
#(10pts) Split the dataset into training (70%) and testing (30%) sets.
#(10pts) Train a Naive Bayes classifier on the training set.
#(10pts) Evaluate the model on the testing set by calculating its accuracy.

#
# def extract_features(text):
#     words = word_tokenize(text.lower())
#     return {word: True for word in words}
#
# x = [d["review"] for d in data]
# y = [d["sentiment"] for d in data]
#
# feature_set = [(extract_features(x),y) for x,y in zip(x,y)]
# train_set = feature_set[:int(len(feature_set)*0.7)]
# test_set =  feature_set[int(len(feature_set)*0.7):]
# print(feature_set)
# bayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
# print(nltk.classify.accuracy(bayes_classifier,test_set))

#TASK 3: Extracting Information from Text/Chunking
#(10pts) Use NLTK to perform chunking to identify noun phrases in the given sentence: "The quick brown fox jumps over the lazy dog.". Define a simple grammar to chunk noun phrases with the pattern:
#Optional determiner (DT)
#Followed by any number of adjectives (JJ)
#Ending with a noun (NN)
#NOTE: Tokenize a sentence and tag words using NLTK before applying the rule.

# sent = "The quick brown fox jumps over the lazy dog."
#
# tokens = word_tokenize(sent)
# tagged = pos_tag(tokens)
#
# grammer = r"NP : {<DT>?<JJ>*<NN>}"
#
# parser = nltk.RegexpParser(grammer)
#
# parsed_chunk = parser.parse(tagged)
#
# print(tagged)
# print("-"*100)
# print(parsed_chunk)









