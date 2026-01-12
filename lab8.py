# **Build an Information Extraction Pipeline:**
#
# **1. Use this text file as input, which contains excerpts from the book "Tvrdjava" by Meša Selimović.**
#
# **2. Divide the raw text into individual sentences to prepare it for further processing.**
#
# **3. Tokenize each sentence into words or tokens for detailed analysis.**
#
# **4. Perform part-of-speech tagging on the tokenized words. So assign grammatical labels (e.g., noun, verb, adjective) to each word in the tokenized sentences.**
#
# **5. Identify and Extract named entities (e.g., person names, organizations, locations, dates).**
#
# **6. Identify relations between entities within sentences, such as:**
#
# **- associations between people and places**
#
# **- Events tied to specific individuals or dates.**
#
# **- Relationships such as "Person X wrote Book Y" or "Event Z occurred in Location L."**
#
# **(e.g., "Sam Altman" is the CEO of "OpenAI," and "OpenAI" operates in "San Francisco" and "London")**
from collections import Counter

import nltk
from nltk.corpus import conll2000
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk, corpus
from nltk.tree import Tree

# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')
#
# #1
# with open('lab8.txt','r',encoding='utf-8') as file:
#     text = file.read()
# #2
# sentences =  sent_tokenize(text)
# #3
# words = [word_tokenize(sentence) for sentence in sentences]
# #4
# pos_tag_sent = [pos_tag(word) for word in words]
# #5
# def ex_named_entities(tagged_sentence):
#     tree = ne_chunk(tagged_sentence)
#     named_entities = []
#     for subtree in tree:
#         if isinstance(subtree, Tree):
#             entity_name = " ".join(c[0] for c in subtree)
#             entity_type = subtree.label()
#             named_entities.append((entity_name,entity_type))
#     return named_entities
#
# named_entities = [ex_named_entities(sentence) for sentence in pos_tag_sent]
# print("named entities :" , named_entities[:5])
# #6
# relations = []
# for sentence_entities in named_entities:
#     for entity1 in sentence_entities:
#         for entity2 in sentence_entities:
#             if entity1 != entity2:
#                 relations.append((entity1,"related to",entity2))
# print(relations)

#2. The IOB format categorizes tagged tokens as I, O, and B. Why are three tags necessary? What problem would be caused if we used I and O tags exclusively?

# tokens = ["New", "York", "University", "student"]
#
# iob_tags = ["B-NP","I-NP","B-NP","I-NP"]
# print("correct tagging")
# for t,tag in zip(tokens,iob_tags):
#     print(t,tag)
#
# io_tags = ["I-NP","I-NP","I-NP","I-NP"]
# print("Incorrect tagging (only I and O)")
# for t,tag in zip(tokens,io_tags):
#     print(t,tag)

#3. Write a tag pattern to match noun phrases containing plural head nouns, e.g., many/JJ researchers/NNS, two/CD weeks/NNS, both/DT new/JJ positions/NNS. Try to do this by generalizing the tag pattern that handled singular noun phrases.

# textchunk = [("many", "JJ"), ("researchers", "NNS"), ("two", "CD"), ("weeks", "NNS"), ("both","DT"), ("new", "JJ"), ("positions", "NNS")]
# parser = nltk.RegexpParser("NP:{<DT>?<CD>?<JJ>*<NNS>}")
# print(parser.parse(textchunk))

#4. Pick one of the three chunk types in the CoNLL-2000 Chunking Corpus. Inspect the data and try to observe any patterns in the POS tag sequences that make up this kind of chunk. Develop a simple chunker using the regular expression chunker nltk.RegexpParser. Discuss any tag sequences that are difficult to chunk reliably.
# nltk.download('conll2000')
# from nltk.corpus import conll2000
#
# for i in range(20):
#     print(i , conll2000.chunked_sents('train.txt',chunk_types=['VP'])[i])
#
# grammer = r"VP: {<[VRMT].*>+}"
# cp = nltk.RegexpParser(grammer)
# for i in range(20):
#     test_sent = (conll2000.chunked_sents('train.txt',chunk_types=['VP'])[i])
#     print(i, print(cp.parse(test_sent)))

#5. An early definition of chunk was the material that occurs between chinks. Develop a chunker that starts by putting the whole sentence in a single chunk, and then does the rest of its work solely by chinking. Determine which tags (or tag sequences) are most likely to make up chinks with the help of your own utility program. Compare the performance and simplicity of this approach relative to a chunker based entirely on chunk rules.
#
# train_sents = conll2000.chunked_sents("train.txt",chunk_types=["NP"])
# chink_tags = Counter()
#
# for sent in train_sents:
#     for word,pos in sent.pos():
#         for subtree in sent.subtrees():
#             if subtree.label() == "NP" and (word,pos) in subtree.leaves():
#                 break
#         else:
#             chink_tags[pos] +=1
#
# print("most common")
# for tag,count in chink_tags.most_common(15):
#     print(f"{tag}:{count}")
#
# grammar = r"""
#     NP: {<.*>+}
#
#     }<IN|VB.*|MD|RB|TO|CC|,|\.>{
# """
#
# chink_chunker = nltk.RegexpParser(grammar)
# test_sent = [("The","DT"), ("quick","JJ"), ("brown","JJ"), ("fox","NN"),
#              ("jumps","VBZ"), ("over","IN"), ("the","DT"), ("lazy","JJ"),
#              ("dog","NN"), (".",".")]
#
# print(chink_chunker.parse(test_sent))
#
# np_grammar = r"""
#     NP: {<DT>?<JJ.*>*<NN.*>+}
#         {<NNP>+}
#         {<PRP>}
# """
# np_chunker = nltk.RegexpParser(np_grammar)
# test_sents = conll2000.chunked_sents("test.txt", chunk_types=["NP"])
#
# print("Chink-only chunker performance:")
# print(chink_chunker.evaluate(test_sents))
#
# print("\nGrammar NP chunker performance:")
# print(np_chunker.evaluate(test_sents))
