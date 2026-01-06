#1 Identify words with multiple pronunciations or meanings, such as "desert or" "bear." Describe how each pronunciation corresponds to different parts of speech.
# from nltk import ConditionalFreqDist
# from nltk.corpus import brown
# from unicodedata import category
#
# brown_news = brown.tagged_words(categories="news")
# cfd = ConditionalFreqDist((word.lower(),tag) for (word,tag) in brown_news)
#
# for word in cfd.conditions():
#     if len(cfd[word]) > 3:
#         print("Ambiguous Words: ",word)
from collections import Counter
from unittest.mock import DEFAULT

import nltk
from nltk import UnigramTagger, word_tokenize, ConditionalFreqDist, BigramTagger, DefaultTagger
from nltk.corpus import brown
from nltk.tag import brill_trainer, brill

# I THINK THIS WAY I WRONG BUT NOT SURE WE WILL COME BACK LATER HERE
#AFTER SEEING TEACHER

# ambiguos = ["desert" , "bear"]
#     for word in ambiguos:
#         nltk.Text(tokens).concordance(word)

#2 Train a unigram tagger on a sample text and test it with a new sentence. Notice any words that aren't tagged.

# training_data = brown.tagged_sents(categories='news')[:500]
# unigram = UnigramTagger(training_data)
#
# test_sentence = word_tokenize("The quick brown fox jumps over the lazy dog.")
# tag = unigram.tag(test_sentence)
# print(tag)

#3. Use the Brown Corpus to search for specific words and tags.
#
# List all distinct words tagged as modals (MD).
# Identify words that serve as both plural nouns and third-person singular verbs (e.g., “deals” or “flies”). (NNS and VBZ).
# Find three-word prepositional phrases in the form: IN + DET + NN (e.g., “in the park”).

# tagged_words = brown.tagged_words()
# # print(brown.tagged_words())
# cfd = ConditionalFreqDist((word.lower(),tag) for word,tag in tagged_words)
# # print(cfd.items())
# print("__"*50)
# modals = {word for word , tag in tagged_words if tag =="MD"}
# # print(modals)
# ambiguous_noun_verb = {word for word in cfd if "NNS" in cfd[word] and "VBZ" in cfd[word]}
#
# # print(brown.tagged_sents())
# sents = []
# for sent in brown.tagged_sents():
#     for i in range(len(sent)-2):
#         if sent[i][1] == "IN" and sent[i+1][1] == "DT" and sent[i+2][1] == "NN":
#             sents.append((sent[i][0],sent[i+1][0],sent[i+2][0]))
# print(sents)
#

#4. Create three combinations of taggers (e.g., unigram, bigram, and default tagger with backoff) and train each on a subset of the Brown Corpus. Measure the accuracy of each combination and compare results. Which combination provides the highest accuracy?
#
# training_data = brown.tagged_sents(categories="news")[:500]
# testing_test = brown.tagged_sents(categories="news")[500:600]
#
# unigram_tagger = UnigramTagger(training_data)
# print(unigram_tagger)
# bigram_tagger = BigramTagger(training_data)
# print(bigram_tagger)
# default_tagger= DefaultTagger("NN")
# print(default_tagger)
#
# print(unigram_tagger.accuracy(testing_test))
# print(bigram_tagger.accuracy(testing_test))
# print(default_tagger.accuracy(testing_test))

#5. Print a table of the counts of distinct words in the Brown Corpus that have 1 to 10 possible tags.

tagged_words = brown.tagged_words()
cfd = ConditionalFreqDist((w.lower(),tag) for w , tag in tagged_words)
#
# tag_count = Counter(len(cfd[word]) for word in cfd)
#
# for i in range(1,11):
#     print(i,"tags: ",tag_count[i])

#6. For the word with the highest tag count, retrieve and print example sentences showing each different tag assignment.
# most_ambiguos_word = max(cfd,key=lambda word:len(cfd[word]))
# tags_for_most_ambiguos_words = cfd[most_ambiguos_word]
#
# print(most_ambiguos_word," tags are : ",list(tags_for_most_ambiguos_words.keys()))
#
# sentence = brown.tagged_sents()
# for tag in tags_for_most_ambiguos_words:
#     for sent in sentence:
#         if(most_ambiguos_word,tag) in [(word.lower(),pos) for word,pos in sent]:
#             print("Tag",tag,":"," ".join(word for word,pos in sent))
#             break

#7. Implement a simple Brill tagger (transformation-based tagging). Start with a unigram tagger as a base and apply rules to correct mistakes.
# training_data = brown.tagged_sents(categories='news')[:500]
# tagger = UnigramTagger(training_data)
# trainer = brill_trainer.BrillTaggerTrainer(tagger,brill.brill24())
# brill_tagger = trainer.train(training_data)
#
# test_data = brown.tagged_sents(categories='news')[500:600]
# brill_accuracy = brill_tagger.accuracy(test_data)
# print("Brill Tagger Accuracy:", brill_accuracy)