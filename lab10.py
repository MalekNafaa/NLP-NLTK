#1. Write a code snippet to create a small vocabulary from a text corpus using the CountVectorizer from sklearn. Tokenize the text and print the resulting vocabulary. You can create a corpus from these sentences: "I love natural language processing.", "Word embeddings are amazing for NLP.", "Machine learning improves text analysis."
# from sklearn.feature_extraction.text import CountVectorizer
#
# corpus = ["I love natural language processing.",
#           "Word embeddings are amazing for NLP.",
#           "Machine learning improves text analysis."]
#
# vectorizer = CountVectorizer()
# tokenized_text = vectorizer.fit_transform(corpus)
# print(vectorizer.vocabulary_)
import matplotlib.pyplot as plt
import numpy as np
from nltk import corpus
from numpy.ma.core import negative
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

#2 Compute the Term Frequency-Inverse Document Frequency (TF-IDF) for the vocabulary using TfidfVectorizer. Display the resulting TF-IDF matrix.

# corpuss = [
#     "The cat sat on the mat",
#     "The dog barked at the cat"
# ]
# tfidf = TfidfVectorizer()
# x = tfidf.fit_transform(corpuss)
# print(x.toarray())
#
#3. Load pretrained Word2Vec embeddings using the gensim library. Use the "Google News" Word2Vec model or any pretrained model of your choice.

import  gensim.downloader as api
model = api.load("glove-wiki-gigaword-50")
#4. Find and display the vector representation for the word "king."
# print(model["king"])
#5
# print(model.similarity("king","queen"))
#6
# print(model.most_similar(positive=["king","queen"],negative=["man"])[0])
#7
# def solve_analogy(word1,word2,words3):
#     return model.most_similar(positive=[word1,word2],negative=[words3])[0]
# # print(solve_analogy("king","queen","man"))
# #8
# words = ["king", "queen", "man", "woman", "child"]
# embedding = np.array([model[word] for word in words])
# #9
# pca = PCA(n_components=2)
# reduced_embedding = pca.fit_transform(embedding)
# print(reduced_embedding)
# #10
# plt.figure()
# for i,word in enumerate(words):
#     plt.scatter(reduced_embedding[i,0],reduced_embedding[i,1])
#     plt.text(reduced_embedding[i,0]+0.01,reduced_embedding[i,0],word)
# plt.title("word embedding visualization")
# plt.show()