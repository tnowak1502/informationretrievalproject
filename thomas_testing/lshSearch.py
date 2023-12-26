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

def lshSearch(docId, analyzer, reader):
    #retrieve the document and its contents and turn them into a token stream for analysis
    doc = reader.storedFields().document(docId)
    contents = doc.get("content")
    ts = analyzer.tokenStream("minHash", contents)
    hashes = set()
    termAtt = ts.addAttribute(CharTermAttribute.class_)
    ts.reset()
    while ts.incrementToken():
        term = termAtt.toString()
        hashes.add(term)
    #print(len(hashes))
    ts.close()
    builder = BooleanQuery.Builder()
    for qt in hashes:
        builder.add(TermQuery(Term("minHash", qt)), BooleanClause.Occur.SHOULD)
    res = builder.build()
    # end = time.time()
    # print("THIRD PART:", end - start)
    return res

def run(searcher, analyzer, reader):
    while True:
        print()
        print("Hit enter with no input to quit.")
        command = input("Query: ")
        if command == '':
            return
        if command == 'gta':
            documents = pd.read_csv("video_games.txt", sep=",")["Sections"]
            command = documents[1000]
        print()
        print("Searching for:", command)
        query = QueryParser("content", analyzer).parse(prune_unwanted(command))
        scoreDocs = searcher.search(query, 5).scoreDocs
        for i in range(len(scoreDocs)):
            scoreDoc = scoreDocs[i]
            doc = searcher.doc(scoreDoc.doc)
            print(i+1, 'title:', doc.get("title"), "| id:", scoreDoc.doc, "|", doc.get("content")[:50])
        print()
        print("Select document to find related documents for by typing its number")
        while True:
            command = input("Select: ")
            if int(command)-1 < 0 or int(command) > len(scoreDocs):
                print("Select a number within range")
            else:
                print()
                break
        lshq = lshSearch(scoreDocs[int(command)-1].doc, analyzer, reader)
        print(lshq)
        likeDocs = searcher.search(lshq, 20).scoreDocs
        print("Relevant documents:")
        for likeDoc in likeDocs:
            doc = searcher.doc(likeDoc.doc)
            print('     title:', doc.get("title"), "| id:", likeDoc.doc)



if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    INDEX_DIR = "indexMinHash.index"
    directory = MMapDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())
    hashMap = HashMap()
    hashMap.put("minHash", MinHashAnalyzer())
    analyzer = PerFieldAnalyzerWrapper(EnglishAnalyzer(), hashMap)
    run(searcher, analyzer, reader)
    del searcher
