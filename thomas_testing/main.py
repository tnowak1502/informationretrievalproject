import numpy
import pandas as pd
import numpy as np
from collections import Counter
import math
import pickle
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import ngrams

def NRA_algo(query_terms, inverted_index, k):
    doc_scores = {}
    doc_worstbest = {}
    lowest_per_term = np.zeros(len(query_terms))
    idx = 0
    while True:
        print("LOOP:", idx, "\n")
        idx_stats = {}
        for i in range(len(query_terms)):
            query_term = query_terms[i][0]
            #idf = query_terms[i][1]
            if idx < len(inverted_index[query_term]):
                doc_id = inverted_index[query_term][idx][0]
                impact = inverted_index[query_term][idx][3]
            else:
                doc_id = -1
                impact = 0
            lowest_per_term[i] = impact
            idx_stats[query_term] = [doc_id, impact, i]
        #print("INDEX STATS:", idx_stats)
        for qt in idx_stats.keys():
            #print(qt)
            stats = idx_stats[qt]
            doc_id = stats[0]
            impact = stats[1]
            i = stats[2]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = numpy.zeros(len(query_terms))
            doc_scores[doc_id][i] = impact
        for doc_id in doc_scores.keys():
            #calculate worst
            doc_worstbest[doc_id] = [np.sum(doc_scores[doc_id]), 0]
            #calculate best
            best = 0
            for i in range(len(doc_scores[doc_id])):
                score = doc_scores[doc_id][i]
                if score == 0:
                    score = lowest_per_term[i]
                best += score
            doc_worstbest[doc_id][1] = best
        idx += 1
        #prune
        #print("DOC WORST BEST PREPRUNE:", doc_worstbest)
        doc_worstbest = dict(sorted(doc_worstbest.items(), key=lambda x: x[1], reverse=True))
        if len(doc_worstbest) > k:
            kth_worst = list(doc_worstbest.values())[k-1][0]
            i = len(doc_worstbest)
            while list(doc_worstbest.values())[len(doc_worstbest)-1][1] < kth_worst:
                del doc_worstbest[list(doc_worstbest.keys())[len(doc_worstbest)-1]]
                #print("DOC WORST BEST:", doc_worstbest)
        #print("DOC SCORES:", doc_scores)
        print("DOC WORST BEST:", doc_worstbest)
        #print("LOWEST PER TERM", lowest_per_term)
        if len(doc_worstbest) <= k:
            break
    return doc_worstbest.items()

def OkapiBm25(query_terms, documents, inverted_index, k, k1=1.2, k3=1.2, b=0.75):
    # Initialize document scores
    doc_scores = {doc_id: 0.0 for doc_id in range(len(documents))}
    # Calculate average document length
    avg_doc_length = 0

    for idx in inverted_index.keys():
        avg_doc_length += sum(doc_length for _, _, doc_length, _ in inverted_index[idx])
    avg_doc_length /= len(documents)

    i = 0
    for query_term, score in query_terms:
        if query_term in inverted_index:
            df = len(inverted_index[query_term])
            idf = math.log(len(documents) / df)

            for doc_id, doc_term_count, doc_length, _ in inverted_index[query_term]:
                tf = (k1 + 1) * doc_term_count / (k1 * ((1 - b) + b * (doc_length / avg_doc_length)) + doc_term_count)
                qt = ((k3 + 1) * qt_count[i]) / k3 + qt_count[i]
                doc_scores[doc_id] += idf * tf * qt
        i += 1

    # Rank documents by their scores
    sorted_documents = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    # Get the top-k documents
    top_k_documents = sorted_documents[:k]
    return top_k_documents

def MixtureModel(query_terms, documents, inverted_index, k, lmb=0.5):
    doc_scores = {doc_id: 0.0 for doc_id in range(len(documents))}
    no_of_terms = 0
    for idx in inverted_index.keys():
        no_of_terms += sum(doc_length for _, _, doc_length, _ in inverted_index[idx])

    for query_term, score in query_terms:
        if query_term in inverted_index:
            cf = sum(doc_term_count for _, doc_term_count, _, _ in inverted_index[query_term])
            for doc_id, doc_term_count, doc_length, _ in inverted_index[query_term]:
                doc_scores[doc_id] += lmb*doc_term_count/doc_length+(1-lmb)*cf/no_of_terms

    # Rank documents by their scores
    sorted_documents = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    # Get the top-k documents
    top_k_documents = sorted_documents[:k]
    return top_k_documents

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
# print(documents.head())
# # Preprocess documents and build the inverted index
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
with open("inverted_index_titles.pkl", "rb") as fp:
    inverted_index = pickle.load(fp)

query = documents[1648]
query = query.replace("[", "")
query = query.replace("]", "")
query = query.replace("\"", "")
query = query.replace("\\n", "")
query = query.replace("\\", "")
query = query.replace("/", " ")
query = query.replace(".", " ")
query = query.replace(",", "")
query = query.replace("-", " ")
query = query.replace("(", "")
query = query.replace(")", "")
query = query.replace(":", " ")
query = query.replace(";", " ")
query = query.replace("'s", "")
#query = query.replace("'", "")

# Preprocess the query
query_terms = [stemmer.stem(term) for term in query.lower().split() if term not in stop_words]
#query_terms += list(ngrams(query_terms, 3))
qt_count = list(Counter(query_terms).values())

query_tf_idf = {}
for query_term, count in Counter(query_terms).items():
    query_term = str(query_term)
    #print(query_term)
    try:
        query_tf_idf[query_term] = count*0.5/np.max(qt_count) * math.log(len(documents) / (len(inverted_index[query_term])))
    except KeyError:
        print("KeyError:", query_term)
        continue
sorted_query_terms = sorted(query_tf_idf.items(), key=lambda x: x[1], reverse=True)
top_10_query_terms = sorted_query_terms[:15]
print(top_10_query_terms)


top_k_documents = OkapiBm25(top_10_query_terms, documents, inverted_index, k=20, k1=1.5, k3=1.0, b=0.75)
#top_k_documents = MixtureModel(top_10_query_terms, documents, inverted_index, k=15, lmb=0.75)
i = 1
for doc_id, score in top_k_documents:
    print(f"{i}.: Document {doc_id}: {documents[doc_id][:300]} (Score: {score:.2f})")
    i+=1

# top_k_documents = NRA_algo(top_10_query_terms, inverted_index, k=5)
# # Print the top-k documents
# i = 1
# for doc_id, scores in top_k_documents:
#     score=scores[0]
#     print(f"{i}.: Document {doc_id}: {documents[doc_id][:300]} (Score: {score:.2f})")
#     i+=1
