#!/usr/bin/env python

INDEX_DIR = "IndexFilesTermVectors.index"
#INDEX_DIR = "IndexFiles.index"
#INDEX_DIR = "IndexFilesShingles.index"

import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
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
def mltByHand(docId, analyzer, reader):
    start = time.time()

    doc = reader.storedFields().document(docId)
    contents = doc.get("contents")
    ts = analyzer.tokenStream("contents", contents)
    qt_count_doc = {}
    termAtt = ts.addAttribute(CharTermAttribute.class_)
    freqAtt = ts.addAttribute(TermFrequencyAttribute.class_)
    ts.reset()
    while(ts.incrementToken()):
        term = termAtt.toString()
        if term in EnglishAnalyzer.ENGLISH_STOP_WORDS_SET:
            continue
        freq = freqAtt.getTermFrequency()
        if qt_count_doc.get(term) is None:
            qt_count_doc[term] = freq
        else:
            qt_count_doc[term] += freq
    ts.close()

    # qt_count_doc = {}
    # qt_count_index = {}
    # terms = reader.getTermVector(docId, "contents")
    # termsEnum = terms.iterator()
    # print(terms.hasPositions())
    # while next(BytesRefIterator(termsEnum)) is not None:
    #     term = termsEnum.term().utf8ToString()
    #     qt_count_doc[term] = termsEnum.docFreq()
    #     qt_count_index[term] = termsEnum.totalTermFreq()
    end = time.time()
    print("FIRST PART:", end - start)
    start = time.time()
    query_tf_idf = {}
    maxCount = max(qt_count_doc.values())
    for query_term, count in qt_count_doc.items():
        # print(query_term)
        try:
            query_tf_idf[query_term] = 0
            if len(query_term) < 9:
                if count > 5:
                    term_freq = reader.docFreq(Term("contents", query_term))#qt_count_index[query_term]
                    if term_freq > 4:
                        query_tf_idf[query_term] = count * 0.5 /maxCount * math.log(
                            21228 / term_freq)
        except KeyError:
            print("KeyError:", query_term)
            continue
    end = time.time()
    print("SECOND PART:", end-start)
    start = time.time()
    sorted_query_terms = sorted(query_tf_idf.items(), key=lambda x: x[1], reverse=True)
    top_10_query_terms = sorted_query_terms[:25]
    res = ""
    for i in range(len(top_10_query_terms)):
        qt = top_10_query_terms[i][0]
        res += "contents:" + qt + " "
    print("res:", res)
    res = QueryParser("contents", analyzer).parse(res)
    end = time.time()
    print("THIRD PART:", end - start)
    return res

def run(searcher, analyzer, reader):
    mlt = MoreLikeThis(reader)
    mlt.setAnalyzer(analyzer)
    mlt.setFieldNames(["contents", "title"])
    #mlt.setMaxWordLen(9)
    #mlt.setMaxDocFreq(1000)
    #mlt.setMinTermFreq(4)
    #mlt.setMaxQueryTerms(20)
    print(mlt.getFieldNames())
    print(mlt.describeParams())
    totalTruth = 0
    totalCorrect = 0
    totalResults = 0
    totalCorrectUnder5 = 0
    for game in groundtruth.keys():
        truth = groundtruth[game]
        totalTruth += len(truth)
        print()
        command = game
        print("Searching for:", command)
        query = QueryParser("contents", analyzer).parse(command)
        scoreDocs = searcher.search(query, 5).scoreDocs
        #print("%s total matching documents." % len(scoreDocs))
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            if doc.get("title") == command:
                print('title:', doc.get("title"), "| id:", scoreDoc.doc)
                #mltq = mlt.like(scoreDoc.doc)
                mltq = mltByHand(scoreDoc.doc, analyzer, reader)
                print("query:\n", mltq)
                likeDocs = searcher.search(mltq, 50).scoreDocs
                print("Like docs:")
                correct = 0
                correctunder5 = 0
                counter = 0
                for likeDoc in likeDocs:
                    doc = searcher.doc(likeDoc.doc)
                    if doc.get("title") in truth:
                        correct += 1
                        if counter < 5:
                            correctunder5 += 1
                    print('     title:', doc.get("title"), "| id:", likeDoc.doc)
                    counter+=1
                print("     correct:", correct, "out of", len(likeDocs), "results and", len(truth), "true answers, with", correctunder5, "in the top 5")
                totalCorrect += correct
                totalCorrectUnder5 += correctunder5
                totalResults += len(likeDocs)
    print("Total correct results:", totalCorrect, "out of a possible", totalTruth, "and", totalResults, "results, with", totalCorrectUnder5, "in the top 5")



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
    run(searcher, analyzer, reader)
    del searcher
