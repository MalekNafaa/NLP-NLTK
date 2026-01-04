from nltk import FreqDist
from nltk.text import Text

from PIL.EpsImagePlugin import field
from nltk.corpus import gutenberg, stopwords

#1.Use the Gutenberg Corpus to explore austen-persuasion.txt. How many word tokens does this book have? How many word types?
# words = gutenberg.words('austen-persuasion.txt')
# print(len(words))
# types = set(w.lower() for w in words)
# print(len(types))

#2.Use the Brown Corpus reader and the Web Text Corpus reader. Your goal is to explore the genres available and access some sample texts.

from nltk.corpus import brown
from nltk.corpus import webtext
#
# print(brown.categories()) # print the genres available in brown corpus
# print(len(brown.categories()))
#
# print([w for w in brown.words(categories='news') if w.isalpha()])
#
# print(webtext.fileids()) # print the fileids available in webtext corpus
# print(len(webtext.fileids()))
#
# print(webtext.words('grail.txt')[10:20])

#3 Read in the texts of the State of the Union addresses, using the state_union corpus reader. Count occurrences of men, women, and people in each document. What has happened to the usage of these words over time?

from nltk.corpus import state_union
from collections import defaultdict
#
# rows=[]
# fileids = state_union.fileids()
# for fid in fileids:
#     year = int(fid[:4])
#     words = [w.lower() for w in state_union.words(fid)]
#     rows.append((year, words.count('men'), words.count('women'), words.count('people')))
# rows.sort()
# print(rows)
# for row in rows:
#     print(row)

#4 We will compare Moby Dick (melville-moby_dick.txt) and Sense and Sensibility (austen-sense.txt) from the Gutenberg corpus.
#A.Load each text. Find the total number of word tokens. Find the total number of unique word types (vocabulary size). Compute the lexical richness = len(tokens) / len(types).
#B.Use text.concordance("monstrous") to explore how the word monstrous is used in both texts. How is the meaning different in Moby Dick versus Sense and Sensibility?

# def print_stats_gutenberg_text(text):
#     words = gutenberg.words(text)
#     tokens = len(words)
#     types = len(set(w.lower() for w in words))
#     richness = tokens / types
#     print(tokens, types , richness)
#
# print_stats_gutenberg_text('melville-moby_dick.txt')
# print_stats_gutenberg_text('austen-sense.txt')
#B
# moby_words = gutenberg.words('melville-moby_dick.txt')
# sense_words = gutenberg.words('austen-sense.txt')
#
# moby_text = Text(moby_words)
# sense_text = Text(sense_words)
#
# moby_text.concordance('monstrous')
# print("moby is done here")
# # sense_text.concordance('monstrous')

#5 Write a program to find all words that occur at least three times in the Brown Corpus.
#
# from nltk.corpus import brown
#
# words = [w.lower() for w in brown.words() if w.isalpha()]
# fdist = FreqDist(words)
# at_least_three = [word for word,count  in fdist.items() if count >=3]
# print(len(at_least_three))

#6 Write a function that finds the 50 most frequently occurring words of a text that are not stopwords.
# from nltk.corpus import brown
#
# stop_words = stopwords.words("english")
# def top_50_no_stopwords(tokens):
#     words = [ w.lower() for w in tokens if w.isalpha() and w.lower() not in stop_words]
#     fd = FreqDist(words)
#     return fd.most_common(50)
#
# print(top_50_no_stopwords(brown.words()))

#7 Write a program to print the 50 most frequent bigrams (pairs of adjacent words) of a text, omitting bigrams that contain stopwords.
#
# from nltk.corpus import  stopwords , brown
# from nltk import bigrams,FreqDist
#
# stopwords = stopwords.words('english')
#
# tokens = [w.lower() for w in brown.words() if w.isalpha()]
#
#
#
# bigrams = [bg for bg in list(bigrams(tokens)) if bg[0] not in stopwords and bg[1] not in stopwords]
#
# fd = FreqDist(bigrams)
# top_50 = fd.most_common(50)
# for (w1,w2),count in top_50:
#     print(w1," ",w2," ",count)

#8 Investigate the holonym-meronym relations for some nouns. Remember that there are three kinds of holonym-meronym relation,
#so you need to use member_meronyms(), part_meronyms(), substance_meronyms(), member_holonyms(), part_holonyms(), and substance_holonyms().

# from nltk.corpus import wordnet as wn
#
# s = wn.synset("car.n.01")
#
# print("Synset: ",s)
# print("Meronyms ",s.member_meronyms())
# print("part_meronyms ", s.part_meronyms())
# print("substance_meronyms", s.substance_meronyms())
# print("Holonyms")
# print("Member holonyms ", s.member_holonyms())
# print("part holonyms ",s.part_holonyms())
# print("substance_holonyms ",s.substance_holonyms())

#9 The CMU Pronouncing Dictionary contains multiple pronunciations for certain words. How many distinct words does it contain? What fraction of words in this dictionary have more than one possible pronunciation?

# from nltk.corpus import cmudict
#
# cmu = cmudict.dict()
# num_distinct_words = len(cmu)
# print(num_distinct_words)
#
# mult = [w for w ,prons in cmu.items() if len(prons)>1]
# fraction = len(mult)/num_distinct_words
#
# print(len(mult))
# print(fraction)

#10 Define a function supergloss(s) that takes a synset s as its argument and returns a string consisting of the concatenation of the definition of s, and the definitions of all the hypernyms and hyponyms of s.

from nltk.corpus import wordnet as wn
def supergloss(s):
    parts = []
    parts.append(s.definition())
    for h in s.hypernyms():
        parts.append(h.definition())
    for hy in s.hyponyms():
        parts.append(hy.definition())
    return " ".join(parts)
s = wn.synset("car.n.01")
print(supergloss(s))

