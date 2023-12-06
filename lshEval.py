#!/usr/bin/env python

import sys, os, lucene, time, math

from java.nio.file import Paths
from java.util import HashMap
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
from org.apache.lucene.index import DirectoryReader, Term, SingleTermsEnum, SlowImpactsEnum
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause, TermQuery
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute, TermFrequencyAttribute, TypeAttribute, PayloadAttribute, BytesTermAttribute
from utils import prune_unwanted
from MinHashAnalyzer import MinHashAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper
import pandas as pd
from lshSearch import lshSearch
import json

groundtruth = json.load(open("groundtruth.json","r"))

def run(searcher, analyzer, reader):
    totalTruth = 0
    totalCorrect = 0
    totalResults = 0
    totalCorrectUnder20 = 0
    for game in groundtruth.keys():
        print(game)
        truth = groundtruth[game]
        game = prune_unwanted(game)
        totalTruth += len(truth)
        query = QueryParser("title", analyzer).parse(prune_unwanted(game))
        scoreDocs = searcher.search(query, 5).scoreDocs
        found = False
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            if prune_unwanted(doc.get("title")) == game:
                found = True
                lshq = lshSearch(scoreDoc.doc, analyzer, reader)
                #print(lshq)
                likeDocs = searcher.search(lshq, 100).scoreDocs
                correct = 0
                correctunder20 = 0
                counter = 0
                for likeDoc in likeDocs:
                    doc = searcher.doc(likeDoc.doc)
                    if doc.get("title") in truth:
                        correct += 1
                        if counter < 20:
                            correctunder20 += 1
                    counter += 1
                prec = correct / len(likeDocs)
                rec = correct / len(truth)
                if prec + rec > 0:
                    f1 = 2 * (prec * rec) / (prec + rec)
                else:
                    f1 = 0
                prec20 = correctunder20 / 20
                totalCorrect += correct
                totalCorrectUnder20 += correctunder20
                totalResults += len(likeDocs)
                print("precision:", prec, "recall:", rec, "f1:", f1, "p@20:", prec20)
                break
            if not found:
                print("Couldn't find", game)
    prec = totalCorrect / totalResults
    rec = totalCorrect / totalTruth
    prec5 = totalCorrectUnder20 / (20 * len(groundtruth.keys()))
    if prec + rec > 0:
        f1 = 2 * (prec * rec) / (prec + rec)
    else:
        f1 = 0
    print("precision:", prec, "recall:", rec, "f1:", f1, "p@20:", prec5)



if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    INDEX_DIR = "indexMinHash.index"
    directory = MMapDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())
    searcher.setMaxClauseCount(20000)
    hashMap = HashMap()
    hashMap.put("minHash", MinHashAnalyzer())
    analyzer = PerFieldAnalyzerWrapper(EnglishAnalyzer(), hashMap)
    run(searcher, analyzer, reader)
    del searcher
