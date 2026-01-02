# Lab Tasks
import nltk
from nltk import FreqDist
from nltk.book import text2, text4, text5, text1, sent1
import matplotlib.pyplot as plt
# 1. Try using the Python interpreter as a calculator, and typing expressions like 12 /(4 + 1).

# print(12 / (4 + 1))



# 2. How many words are there in text2? How many distinct words are there?

# print(len(text2))
# print("distinct words",len(set(text2)),"\n")


# 3. Produce a dispersion plot of the four main protagonists in Sense and Sensibility: Elinor, Marianne, Edward, and Willoughby.
# What can you observe about the different roles played by the males and females in this novel? Can you identify the couples?
# Hint: Sense and Sensibility by Jane Austen is located in the nltk’s book module, namely the text2.

# text2.dispersion_plot(['Elinor', 'Marianne', 'Edward', 'Willoughby'])
# plt.show()

# 4. Consider the following Python expression: len(set(text4)). State the purpose of this expression. Describe the two steps involved in performing this computation.

# print(len(set(text4)))

# 5. Find the collocations in text5.

# text5.collocations()

# 6. Find 5 the most common words in text1 that are longer than 5 letters. You should print the most common words and plot the results.

# fdist = FreqDist(w.lower() for w in text1 if len(w) > 5 and w.isalpha())
# print(fdist.most_common(5))
# fdist.plot(5)
# plt.show()

# 7. Write the slice expression that extracts the last two words of text2.

# print(text2[-2:])

# Basic Tasks
#
# 1. Given an alphabet of 26 letters, there are 26 to the power 10, or 26 ** 10, 10- letter strings we can form. That works out to 141167095653376L (the L at the end just indicates that this is Python’s long-number format). How many hundred-letter strings are possible?

# num_1 = 26 ** 100
# print(num_1)
# print("hundred-letter strings are possible",len(str(num_1)))

# 2. The Python multiplication operation can be applied to lists. What happens when you type ['Monty', 'Python'] * 20, or 3 * sent1?

# print(['Monty', 'Python'] * 20)
# print(3 * sent1)

# 3. Define a string and assign it to a variable, e.g., my_string = "robots explore Mars while astronauts dream of distant galaxies". Try adding the string to itself using my_string + my_string, or multiplying it by a number, e.g., my_string * 3. Notice that the strings are joined together without any spaces. How could you fix this?

# my_string = "robots explore Mars while astronauts dream of distant galaxies"
# print(my_string + my_string)
# print(my_string * 3)
# # Fix
# print(my_string + " " + my_string)
# print(" ".join(my_string))

# 4. Define a variable my_sent to be a list of words, using the syntax my_sent = ["rocket", "planet", "galaxy", "astronaut", "blackhole"].

# my_sent = ["rocket", "planet", "galaxy", "astronaut", "blackhole"]
#
# # a. Convert this list into a single string where the words are separated by spaces.
# single_string = " ".join(my_sent)
# print(single_string)
# # b. Split the string back into the list form you had to start with.
# split_list = single_string.split()
# print(split_list)

# 5. Define several variables containing lists of words, e.g., phrase1, phrase2. Join them together in various combinations (using the plus operator) to form whole sentences. What is the relationship between len(phrase1 + phrase2) and len(phrase1) + len(phrase2)?

# phrase1 = ["I", "love", "natural"]
# phrase2 = ["language", "processing"]
#
# combined = phrase1 + phrase2
#
# print(combined)
# print(len(combined))
# print(len(phrase1) + len(phrase2))


# 6. Consider the following two expressions, which have the same value. Which one will typically be more relevant in NLP? Why?
#  ["Monty Python"[6:12]
#  ["Monty", "Python"][1]



# 7. We have seen how to represent a sentence as a list of words, where each word is a sequence of characters. What does sent1[2][2] do? Why? Experiment with other index values.
# 8. The first sentence of text3 is provided to you in the variable sent3. The index of 'the' in sent3 is 1, because sent3[1] gives us 'the'. What are the indexes of the two other occurrences of this word in sent3?
#
# 9. Find all words in the Chat Corpus (text5) starting with the letter b. Show them in alphabetical order.
# 10. Write a sentence and then convert all letters to uppercase?
# 11. Write your name and make all letters to be lowercase.
# 12. Show the text1 richness. You should also find and count the most common words from it.
# 13. Find the longest and the shortest word in text4.

#
# Tasks
# Intermediate Tasks
# 1. Use text9.index() to find the index of the word sunset. You’ll need to insert this word as an argument between the parentheses. By a process of trial and error, find the slice for the complete sentence that contains this word.
# 2. Using list addition, and the set and sorted operations, compute the vocabulary of the sentences sent1 ... sent5.
# 3. There are two different ways to produce a sorted vocabulary list from text1 after converting words to lowercase and removing duplicates. In one approach, you lowercase all the words first and then remove duplicates. In the other approach, you remove duplicates first and then lowercase the remaining words.
# a. Write both Python expressions, one for each approach.
# b. Compare the results of the two expressions. Which one gives a larger vocabulary size? Why?
# c. Will this difference always occur with other texts, or does it depend on the data?
# 4. Find all the four-letter words in the Chat Corpus (text5). With the help of a frequency distribution (FreqDist), show these words in decreasing order of frequency.
# 5. Use a combination of for and if statements to loop over the words of the movie script for Monty Python and the Holy Grail (text6) and print all the uppercase words, one per line.
# 6. Write expressions for finding all words in text6 that meet the following conditions. The result should be in the form of a list of words: ['word1', 'word2', ...].
# a. Ending in ize
# b. Containing the letter z
# c. Containing the sequence of letters pt
# d. All lowercase letters except for an initial capital (i.e., titlecase)
# 7. Define sent to be the list of words ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']. Now write code to perform the following tasks:
# a. Print all words beginning with sh.
#
#
# b. Print all words longer than four characters
# 8. Write a Python expression that calculates the total number of characters in all the words of text1.
#  a. Now extend your solution to compute the average word length in text1.


# 9. Define a function called vocab_size(text) that has a single parameter for the text, and which returns the vocabulary size of the text.


# 10. Define a function percent(word, text) that calculates how often a given word occurs in a text and expresses the result as a percentage.


# 11. We have been using sets to store vocabularies. Try the following Python expression:
# set(sent3) < set(text1).
# Experiment with this using different arguments to set(). What does it do? Can you think of a practical application for this?