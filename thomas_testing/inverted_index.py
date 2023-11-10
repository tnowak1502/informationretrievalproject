from collections import Counter
import math

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams
import pandas as pd
import pickle
import json

documents = pd.read_csv("video_games.txt", sep=",")
documents["both"] = documents["Title"] + " " + documents["Sections"]
documents = documents["both"]
for i in range(len(documents)):
    documents[i] = documents[i].replace("[", "")
    documents[i] = documents[i].replace("]", "")
    documents[i] = documents[i].replace("\"", "")
    documents[i] = documents[i].replace("\\n", "")
    documents[i] = documents[i].replace("\\", "")
    documents[i] = documents[i].replace("/", " ")
    documents[i] = documents[i].replace(".", " ")
    documents[i] = documents[i].replace(",", " ")
    documents[i] = documents[i].replace("-", " ")
    documents[i] = documents[i].replace("(", "")
    documents[i] = documents[i].replace(")", "")
    documents[i] = documents[i].replace(":", " ")
    documents[i] = documents[i].replace(";", " ")
    documents[i] = documents[i].replace("'s", "")
    #documents[i] = documents[i].replace("'", "")
inverted_index = {}

# Preprocess documents and build the inverted index
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

print(len(documents))
for doc_id, document in enumerate(documents):
    print(doc_id)
    terms = [stemmer.stem(term) for term in document.lower().split() if term not in stop_words]
    term_count = Counter(terms)
    doc_length = len(terms)

    for term, count in term_count.items():
        term = str(term)
        # impact = (1+math.log(count))/doc_length
        # #if term == "game": print("doc", doc_id, count, doc_length, impact)
        # if term in inverted_index:
        #     idx = len(inverted_index[term])
        #     for i in range(len(inverted_index[term])):
        #         #if term == "game": print("other", inverted_index[term][i])
        #         if inverted_index[term][i][3] < impact:
        #             idx = i
        #             #if term == "game": print("break", idx)
        #             break
        #     inverted_index[term].insert(idx, (doc_id, count, doc_length, impact))
        #     #if term == "game": print(inverted_index[term])
        # else:
        #     inverted_index[term] = [(doc_id, count, doc_length, impact)]
        if term in inverted_index:
            inverted_index[term].append((doc_id, count, doc_length, 0))
        else:
            inverted_index[term] = [(doc_id, count, doc_length, 0)]

with open("inverted_index_titles.pkl", "wb") as fp:
    pickle.dump(inverted_index, fp)
