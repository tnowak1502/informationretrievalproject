#!/usr/bin/env python

#INDEX_DIR = "IndexFilesTermVectors.index"
#INDEX_DIR = "IndexFiles.index"
#INDEX_DIR = "IndexFilesShingles.index"
INDEX_DIR = "index.index"

import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
from org.apache.lucene.index import DirectoryReader, IndexReader, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.queries.mlt import MoreLikeThis
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute, TermFrequencyAttribute
import pandas as pd
import numpy as np
import math
import pickle
import json
import nltk
from nltk.corpus import stopwords
#nltk.download("stopwords")
from nltk.stem import PorterStemmer
from collections import Counter
import time
from mltSearch import mltByHand

groundtruth = json.load(open("groundtruth.json","r"))
#with open("inverted_index_titles.pkl", "rb") as fp:
#    inverted_index = pickle.load(fp)
#stop_words = set(stopwords.words('english'))
#stemmer = PorterStemmer()

"""
This script is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it
will search the Lucene index in the current directory called 'index' for the
search query entered against the 'contents' field.  It will then display the
'path' and 'name' fields for each of the hits it finds in the index.  Note that
search.close() is currently commented out because it causes a stack overflow in
some cases.
"""

def run(searcher, analyzer, reader, maxLens, minDocFreqs, minTermFreqs):
    best_prec = {"maxLen": 0, "minDocFreq": 0, "minTermFreq": 0, "val": 0}
    best_rec = {"maxLen": 0, "minDocFreq": 0, "minTermFreq": 0, "val": 0}
    best_f1 = {"maxLen": 0, "minDocFreq": 0, "minTermFreq": 0, "val": 0}
    best_prec5 = {"maxLen": 0, "minDocFreq": 0, "minTermFreq": 0, "val": 0}
    for maxLen in maxLens:
        for minDocFreq in minDocFreqs:
            for minTermFreq in minTermFreqs:
                print("maxLen:", maxLen, "minDocFreq:", minDocFreq, "minTermFreq:", minTermFreq)
                totalTruth = 0
                totalCorrect = 0
                totalResults = 0
                totalCorrectUnder5 = 0
                for game in groundtruth.keys():
                    truth = groundtruth[game]
                    totalTruth += len(truth)
                    #print()
                    command = game
                    #print("Searching for:", command)
                    query = QueryParser("content", analyzer).parse(command)
                    scoreDocs = searcher.search(query, 5).scoreDocs
                    #print("%s total matching documents." % len(scoreDocs))
                    for scoreDoc in scoreDocs:
                        doc = searcher.doc(scoreDoc.doc)
                        if doc.get("title") == command:
                            #print('title:', doc.get("title"), "| id:", scoreDoc.doc)
                            #mltq = mlt.like(scoreDoc.doc)
                            mltq = mltByHand(scoreDoc.doc, analyzer, reader, maxLen, minDocFreq, minTermFreq)
                            #print("query:\n", mltq)
                            likeDocs = searcher.search(mltq, 20).scoreDocs
                            #print("Like docs:")
                            correct = 0
                            correctunder5 = 0
                            counter = 0
                            for likeDoc in likeDocs:
                                doc = searcher.doc(likeDoc.doc)
                                if doc.get("title") in truth:
                                    correct += 1
                                    if counter < 5:
                                        correctunder5 += 1
                                #print('     title:', doc.get("title"), "| id:", likeDoc.doc)
                                counter+=1
                            #print("     correct:", correct, "out of", len(likeDocs), "results and", len(truth), "true answers, with", correctunder5, "in the top 5")
                            prec = correct/len(likeDocs)
                            rec = correct/len(truth)
                            if prec+rec > 0:
                                f1 = 2*(prec*rec)/(prec+rec)
                            else:
                                f1 = 0
                            prec5 = correctunder5/5
                            #print("     precision:", prec, "recall:", rec, "f1:", f1, "p@5:", prec5)
                            totalCorrect += correct
                            totalCorrectUnder5 += correctunder5
                            totalResults += len(likeDocs)
                #print("Total correct results:", totalCorrect, "out of a possible", totalTruth, "and", totalResults, "results, with", totalCorrectUnder5, "in the top 5")
                prec = totalCorrect/totalResults
                rec = totalCorrect/totalTruth
                prec5 = totalCorrectUnder5/(5*len(groundtruth.keys()))
                if prec+rec > 0:
                    f1 = 2*(prec*rec)/(prec+rec)
                else:
                    f1 = 0
                if prec > best_prec["val"]:
                    best_prec["val"] = prec
                    best_prec["maxLen"] = maxLen
                    best_prec["minDocFreq"] = minDocFreq
                    best_prec["minTermFreq"] = minTermFreq
                if rec > best_rec["val"]:
                    best_rec["val"] = rec
                    best_rec["maxLen"] = maxLen
                    best_rec["minDocFreq"] = minDocFreq
                    best_rec["minTermFreq"] = minTermFreq
                if f1 > best_f1["val"]:
                    best_f1["val"] = f1
                    best_f1["maxLen"] = maxLen
                    best_f1["minDocFreq"] = minDocFreq
                    best_f1["minTermFreq"] = minTermFreq
                if prec5 > best_prec5["val"]:
                    best_prec5["val"] = prec5
                    best_prec5["maxLen"] = maxLen
                    best_prec5["minDocFreq"] = minDocFreq
                    best_prec5["minTermFreq"] = minTermFreq
                print("precision:", prec, "recall:", rec, "f1:", f1, "p@5:", prec5)
    print(best_prec)
    print(best_rec)
    print(best_f1)
    print(best_prec5)

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    directory = MMapDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())
    analyzer = EnglishAnalyzer()
    #analyzer = ShingleAnalyzerWrapper(analyzer, 3)
    run(searcher, analyzer, reader, maxLens=[5, 6, 7, 8, 9, 10], minDocFreqs=[0, 1, 2, 3, 4, 5], minTermFreqs=[0, 1, 2, 3, 4, 5])
    del searcher
