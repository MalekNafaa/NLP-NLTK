#1 Define a string s = 'colorless'. Write a Python statement that changes this to “colourless” using only the slice and concatenation operations

# s = "colorless"
# s2 = s[:4]+"u"+s[4:]
# print(s2)

#2 Describe the class of strings matched by the following regular expressions:
# [a-zA-Z]+ letters only!!
# [A-Z][a-z]*  first letter capital then any letter (only letters again)
# p[aeiou]{,2}t first letter must be p, and we can have 0 to 2 vowels  and ends in a t.
# \d+(\.\d+)?
# ([^aeiou][aeiou][^aeiou])*
# \w+|[^\w\s]+


#3 Read in some text from a corpus, tokenize it, and print the list of all wh-word types that occur.
# (wh-words in English are used in questions, relative clauses, and exclamations: who, which, what, and so on.) Print them in order. Are any words duplicated in this list, because of the presence of case distinctions or punctuation?

from nltk.corpus import brown
import re

# tokens =[w.lower() for w in brown.words(categories="news") if re.search(r"[a-zA-Z]",w)]
#
# wh_words = [wh for wh in tokens if wh.startswith("wh")]
# wh_types = sorted(set(wh_words))
# print(wh_types)

#4 Write a function unknown() that takes a URL as its argument, and returns a list of unknown words that occur on that web page.
#In order to do this, extract all substrings consisting of lowercase letters (using re.findall()) and remove any items from this set that occur in the Words Corpus (nltk.corpus.words). Try to categorize these words manually and discuss your findings.

# import requests
# from nltk.corpus import words as nltk_words
#
# en_words = [w.lower() for w in nltk_words.words()]
#
# def unknown(url):
#     html = requests.get(url,timeout=15).text
#     tokens = re.findall(r"[a-z]+",html)
#
#     unknown_wrods = sorted(set(w for w in tokens if w not in en_words))
#     return unknown_wrods
# print(unknown("https://www.gutenberg.org/files/2554/2554-0.txt"[:100]))

#5 Fetch and Process Text from a Web Page
# Fetch the Text - Use a URL (choose an interesting article, like “Artificial Intelligence”, or a news article)
# Extract and Clean Text
# Use BeautifulSoup to remove HTML tags.
# Remove special characters
# Tokenize the cleaned text into individual words.
# Stemming and Lemmatization:
# Perform stemming using NLTK's PorterStemmer.
# Perform lemmatization using NLTK's WordNetLemmatizer.
#
# Use regular expressions to find and extract words starting with "a" and "t" (case-insensitive).
# Use another regex pattern to extract all the words that are 5 characters long.
# Convert the text to lowercase and uppercase.
# Find the longest word in the text.
# Find the m
# Most commonly used word in the text.

import re
import requests
from bs4 import BeautifulSoup

import nltk
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
#
# url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
#
# html = requests.get(url,timeout=15).text
#
# soup = BeautifulSoup(html,"html.parser")
# raw_text = soup.get_text(separator=" ")
#
# clean_text = re.sub(r"[^A-Za-z\s]"," ",raw_text)
# clean_text = re.sub(r"\s+"," ",clean_text).strip()
#
# tokens = word_tokenize(clean_text)
#
# tokens = [ t for t in tokens if t.isalpha()]
#
# stemmer = PorterStemmer()
# stems = [stemmer.stem(t.lower()) for t in tokens]
#
# lemmatizer = WordNetLemmatizer()
# lemmas = [lemmatizer.lemmatize(t.lower())for t in tokens]
#
# a_or_t = re.findall(r"\b[atAt][a-zA-Z]*\b",clean_text)
# five_letters = re.findall(r"\b[a-zA-Z]{5}\b", clean_text)
#
# upper = clean_text.upper()
# lower = clean_text.lower()
#
# longest = max(tokens,key=len) if tokens else None
#
# fd = FreqDist([t.lower() for t in tokens])
# most_common_word, most_common_count = fd.most_common(1)[0] if len(fd) else (None, 0)
#
# print("URL:", url)
# print("Number of tokens:", len(tokens))
# print("Sample tokens:", tokens[:20])
#
# print("\nWords starting with a or t (sample):", a_or_t[:20])
# print("5-letter words (sample):", five_letters[:20])
#
# print("\nLongest word:", longest)
# print("Most common word:", most_common_word, "count:", most_common_count)
#
# print("\nStemming sample:", list(zip(tokens[:10], stems[:10])))
# print("Lemmatization sample:", list(zip(tokens[:10], lemmas[:10])))

#6 Text Analysis for a given text. You can use this text for the following tasks:
# Text Extraction and Cleaning - clean up special characters, if needed
# Tokenize the text into individual words.
# Perform stemming on each token.
# Perform lemmatization on each token.
# Find and extract all words that start with "n" or "p" (case-insensitive).
# Extract all words that have exactly 4 characters.
# Convert the text to lowercase and uppercase.
# Find the longest word in the text.
# Find the most commonly used word in the text.

# text = "Natural Language Processing (NLP) is a branch of artificial intelligence focused on the interaction between computers and human language. NLP involves processing and analyzing large amounts of natural language data to enable machines to understand human language in a valuable way."
#
# clean_text = re.sub(r"[^a-zA-Z/s]"," ",text)
# print(clean_text)
# clean_text = re.sub(r"\s+"," ",clean_text).strip()
#
# tokens = word_tokenize(clean_text)
#
# #stemming
# stemmer = PorterStemmer()
# stems = [stemmer.stem(t.lower()) for t in tokens]
# print("stemmed: ",list(zip(tokens,stems)))
#
# #lemmatization
# lemmatizer = WordNetLemmatizer()
# lemmas = [lemmatizer.lemmatize(t.lower()) for t in tokens]
#
# print("Lemmatized: ",list(zip(tokens,lemmas)))
#
# np_words = [t for t in tokens if t.lower().startswith(("n","p"))]
# print(np_words)
#
# four_letters = [t.lower() for t in tokens if len(t) ==4]
# print(four_letters)
#
# print(clean_text.lower())
# print(clean_text.upper())
#
# longest = max(tokens,key=len)
# print(longest)
#
# fd = FreqDist([t.lower() for t in tokens])
# most_common_word = fd.most_common(1)
# print(most_common_word[0][0])

#Intermediate Tasks
#1 Create a file consisting of words and (made up) frequencies, where each line consists of a word, the space character, and a positive integer, e.g., fuzzy 53. Read the file into a Python list using open(filename).readlines(). Next, break each line into its two fields using split(), and convert the number into an integer using int(). The result should be a list of the form: [['fuzzy', 53], ...].

# lines = [
#     "fuzzy 53\n",
#     "nlp 120\n",
#     "python 77\n",
#     "token 15\n"
# ]
#
# with open("freqs.txt","w",encoding="utf-8") as f:
#     f.writelines(lines)
#
# data=[]
#
# for line in open("freqs.txt","r",encoding="utf-8").readlines():
#     word,num = line.split()
#     data.append([word,int(num)])
# print(data)

