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
from utils import prune_unwanted
import json
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

def run(searcher, analyzer, reader, maxLens, minTfs, minDocFreqs, noOfTermss):
    best_prec = {"maxLen": 0, "minTf": 0, "minDocFreq": 0, "noOfTerms": 0, "val": 0}
    best_rec = {"maxLen": 0, "minTf": 0, "minDocFreq": 0, "noOfTerms": 0, "val": 0}
    best_f1 = {"maxLen": 0, "minTf": 0, "minDocFreq": 0, "noOfTerms": 0, "val": 0}
    best_prec20 = {"maxLen": 0, "minTf": 0, "minDocFreq": 0, "noOfTerms": 0, "val": 0}
    for maxLen in maxLens:
        for minTf in minTfs:
            for minDocFreq in minDocFreqs:
                for noOfTerms in noOfTermss:
                    print("maxLen:", maxLen, "minTf:", minTf, "minDocFreq:", minDocFreq, "noOfTerms:", noOfTerms)
                    totalTruth = 0
                    totalCorrect = 0
                    totalResults = 0
                    totalCorrectUnder20 = 0
                    for game in groundtruth.keys():
                        print(game)
                        truth = groundtruth[game]
                        game = prune_unwanted(game)
                        totalTruth += len(truth)
                        query = QueryParser("title", analyzer).parse(game)
                        scoreDocs = searcher.search(query, 10).scoreDocs
                        found = False
                        for scoreDoc in scoreDocs:
                            doc = searcher.doc(scoreDoc.doc)
                            if prune_unwanted(doc.get("title")) == game:
                                found = True
                                mltq = mltByHand(scoreDoc.doc, analyzer, reader, maxLen, minTf, minDocFreq, noOfTerms)
                                likeDocs = searcher.search(mltq, 100).scoreDocs
                                correct = 0
                                correctunder20 = 0
                                counter = 0
                                for likeDoc in likeDocs:
                                    doc = searcher.doc(likeDoc.doc)
                                    if doc.get("title") in truth:
                                        correct += 1
                                        if counter < 20:
                                            correctunder20 += 1
                                    counter+=1
                                prec = correct/len(likeDocs)
                                rec = correct/len(truth)
                                if prec+rec > 0:
                                    f1 = 2*(prec*rec)/(prec+rec)
                                else:
                                    f1 = 0
                                prec20 = correctunder20/20
                                totalCorrect += correct
                                totalCorrectUnder20 += correctunder20
                                totalResults += len(likeDocs)
                                print("precision:", prec, "recall:", rec, "f1:", f1, "p@20:", prec20)
                                break
                        if not found:
                            print("Couldn't find", game)
                    #print("Total correct results:", totalCorrect, "out of a possible", totalTruth, "and", totalResults, "results, with", totalCorrectUnder5, "in the top 5")
                    prec = totalCorrect/totalResults
                    rec = totalCorrect/totalTruth
                    prec20 = totalCorrectUnder20/(20*len(groundtruth.keys()))
                    if prec+rec > 0:
                        f1 = 2*(prec*rec)/(prec+rec)
                    else:
                        f1 = 0
                    if prec > best_prec["val"]:
                        best_prec["val"] = prec
                        best_prec["maxLen"] = maxLen
                        best_prec["minTf"] = minTf
                        best_prec["minDocFreq"] = minDocFreq
                        best_prec["noOfTerms"] = noOfTerms
                    if rec > best_rec["val"]:
                        best_rec["val"] = rec
                        best_rec["maxLen"] = maxLen
                        best_rec["minTf"] = minTf
                        best_rec["minDocFreq"] = minDocFreq
                        best_rec["noOfTerms"] = noOfTerms
                    if f1 > best_f1["val"]:
                        best_f1["val"] = f1
                        best_f1["maxLen"] = maxLen
                        best_f1["minTf"] = minTf
                        best_f1["minDocFreq"] = minDocFreq
                        best_f1["noOfTerms"] = noOfTerms
                    if prec20 > best_prec20["val"]:
                        best_prec20["val"] = prec20
                        best_prec20["maxLen"] = maxLen
                        best_prec20["minTf"] = minTf
                        best_prec20["minDocFreq"] = minDocFreq
                        best_prec20["noOfTerms"] = noOfTerms
                    print("precision:", prec, "recall:", rec, "f1:", f1, "p@20:", prec20)
    print("PRECISION", best_prec)
    print("RECALL", best_rec)
    print("F1", best_f1)
    print("PREC@20", best_prec20)

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
    run(searcher, analyzer, reader, maxLens=[9], minTfs=[4], minDocFreqs=[2], noOfTermss = [75])
    del searcher
