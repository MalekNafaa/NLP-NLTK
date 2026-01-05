import matplotlib.pyplot as plt
from joblib.externals.loky.backend.synchronize import Condition
from nltk import FreqDist, ConditionalFreqDist, download, pos_tag
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

# from nltk.corpus import wordnet as wn
# def supergloss(s):
#     parts = []
#     parts.append(s.definition())
#     for h in s.hypernyms():
#         parts.append(h.definition())
#     for hy in s.hyponyms():
#         parts.append(hy.definition())
#     return " ".join(parts)
# s = wn.synset("car.n.01")
# print(supergloss(s))


#1 Define a conditional frequency distribution over the Names Corpus that allows you to see which initial letters are more frequent for males versus females.

# import nltk
# nltk.download("names")
# from nltk.corpus import names
#
# cfd = ConditionalFreqDist(
#     (fileid[:-4],name[0].lower())
#     for fileid in names.fileids()
#     for name in names.words(fileid)
# )
# print(cfd["female"].most_common(10))
# print(cfd["male"].most_common(10))
# cfd.plot()
# plt.show()

#2 Pick a pair of texts and study the differences between them, in terms of vocabulary, vocabulary richness, genre, etc. Can you find pairs of words that have quite different meanings across the two texts, such as monstrous in Moby Dick and in Sense and Sensibility?

# moby = [ w.lower() for w in  gutenberg.words("melville-moby_dick.txt") if w.isalpha()]
# sense = [w.lower() for w in gutenberg.words("austen-sense.txt") if w.isalpha()]
#
# def stats(tokens):
#     num_tokens = len(tokens)
#     num_types = len(set(tokens))
#     rich = num_tokens / num_types
#     print(num_tokens)
#     print(num_types)
#     print(rich)
# stats(moby)
# print(".....")
# stats(sense)

# #3  What percentage of noun synsets have no hyponyms? You can get all noun synsets using wn.all_synsets('n').
#
# import nltk
# nltk.download("wordnet")
# from nltk.corpus import wordnet as wn
#
# noun_synsets = list(wn.all_synsets('n'))
# total = len(noun_synsets)
#
# no_hyponyms = [ h for h in noun_synsets if len(h.hyponyms()) == 0]
#
# print("percentage = ",(100*len(no_hyponyms)/total))

#4 Write a program to generate a table of lexical diversity scores (i.e., token/type ratios). Include the full set of Brown Corpus genres (nltk.corpus.brown.categories()).
# Which genre has the lowest diversity (greatest number of tokens per type)? Is this what you would have expected?

# from nltk.corpus import brown
#
# def token_type_ratio(tokens):
#     tokens = [w.lower() for w in tokens if w.isalpha()]
#     return len(tokens)/len(set(tokens))
#
# rows = []
# for genre in brown.categories():
#     words = brown.words(categories=genre)
#     ttr = token_type_ratio(words)
#     rows.append((genre,ttr,len(words)))
#
# rows.sort(key= lambda x:x[1],reverse=True)
#
# for g,r,c in rows:
#     print(f"{g:12} | {r:.2f}")
#
# print("lowest diversity :",rows[0][0])

#5  Write a function word_freq() that takes a word and the name of a section of the Brown Corpus as arguments, and computes the frequency of the word in that section of the corpus.

# def word_freq(word,name):
#     tokens = [w.lower() for w in brown.words(categories=name) if w.isalpha()]
#     return tokens.count(word)/len(tokens)
# print("Freq of 'government' in news:", word_freq("government", "news"))
# print("Freq of 'love' in romance:", word_freq("love", "romance"))

#6 Write a program to guess the number of syllables contained in a text, making use of the CMU Pronouncing Dictionary.

# from nltk.corpus import cmudict
# cmu = cmudict.dict()
#
#
# def count_syllables_in_word(word):
#     word = word.lower()
#     if word not in cmu:
#         return 0
#
#     # Take first pronunciation list
#     pron = cmu[word][0]
#
#     # Count vowel phonemes (they end with a digit)
#     return sum(1 for p in pron if p[-1].isdigit())
#
#
# def estimate_syllables(tokens):
#     total = 0
#     unkown =0
#
#     for w in tokens:
#         w = "".join(ch for ch in w if ch.isalpha())
#         if not w:
#             continue
#     s = count_syllables_in_word(w)
#     if s == 0:
#         unkown +=1
#     total +=s
#     return total , unkown
#
# tokens = brown.words(categories="news")[:2000]   # sample first 2000 tokens to keep it fast
# total_syl, unknown_count = estimate_syllables(tokens)
#
# print("Estimated syllables:", total_syl)
# print("Unknown words (not in CMU dict):", unknown_count)

#7 Define a function hedge(text) that processes a text and produces a new version with the word 'like' between every third word.
#
# def hedge(text):
#     result = []
#
#     for  i , w  in enumerate(text,start=1):
#         result.append(w)
#         if i % 3 == 0:
#             result.append("like")
#     return result
# sample = ["natural", "language", "processing", "is", "fun", "today"]
# print("Hedged:", hedge(sample))

#Advanced Tasks
#1 Define a function find_language() that takes a string as its argument and returns a list of languages that have that string as a word. Use the udhr corpus and limit your searches to files in the Latin-1 encoding.
#
# import nltk
# nltk.download("udhr")
# from nltk.corpus import udhr
#
# def find_language(word):
#     word = word.lower()
#     languages = []
#     for fid in udhr.fileids():
#         if not fid.endswith("-Latin1"):
#             continue
#         tokens = [w.lower() for w in udhr.words(fid)]
#         if word in tokens:
#             languages.append(fid[:-7])
#     return sorted(languages)
# print(find_language("freedom"))

# 2 What is the branching factor of the noun hypernym hierarchy? I.e., for every noun synset that has hyponyms—or children in the hypernym hierarchy—how many do they have on average? You can get all noun synsets using wn.all_synsets('n').

from nltk.corpus import wordnet as wn
#
# noun_sysnet = list(wn.all_synsets('n'))
# children = []
#
# for s in noun_sysnet:
#     kids = s.hypernyms()
#     if kids:
#         children.append(len(kids))
# avg_branching = sum(children)/len(children)
#
# print(len(noun_sysnet))
# print(len(children))
# print(avg_branching)

#3 The polysemy of a word is the number of senses it has. Using WordNet, we can determine that the noun dog has seven senses with len(wn.synsets('dog', 'n')). Compute the average polysemy of nouns, verbs, adjectives, and adverbs according to WordNet.


def average_polysemy(pos_tag):
    lemmas = wn.all_lemma_names(pos=pos_tag)
    lemmas = list(lemmas)

    total_senses = 0
    for w in lemmas:
        total_senses += len(wn.synsets(w,pos=pos_tag))

    return total_senses/len(lemmas)

avg_n = average_polysemy('n')
avg_v = average_polysemy('v')
avg_a = average_polysemy('a')  # adjective
avg_r = average_polysemy('r')  # adverb

print("Average polysemy (nouns):", avg_n)
print("Average polysemy (verbs):", avg_v)
print("Average polysemy (adjectives):", avg_a)
print("Average polysemy (adverbs):", avg_r)