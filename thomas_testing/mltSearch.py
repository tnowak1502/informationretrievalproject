#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.queries.mlt import MoreLikeThis
import pandas as pd

"""
This script is loosely based on the Lucene (java implementation) demo class
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it
will search the Lucene index in the current directory called 'index' for the
search query entered against the 'contents' field.  It will then display the
'path' and 'name' fields for each of the hits it finds in the index.  Note that
search.close() is currently commented out because it causes a stack overflow in
some cases.
"""
def run(searcher, analyzer, reader):
    mlt = MoreLikeThis(reader)
    mlt.setAnalyzer(analyzer)
    mlt.setFieldNames(["contents", "title"])
    mlt.setMaxWordLen(20)
    print(mlt.getFieldNames())
    print(mlt.describeParams())
    while True:
        print()
        print("Hit enter with no input to quit.")
        command = input("Query:")
        if command == '':
            return
        print()
        print("Searching for:", command)
        query = QueryParser("contents", analyzer).parse(command)
        scoreDocs = searcher.search(query, 1).scoreDocs
        #print("%s total matching documents." % len(scoreDocs))

        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            fields = doc.getFields()
            print('title:', doc.get("title"), ", id:", scoreDoc.doc)
            mltq = mlt.like(scoreDoc.doc)
            print("query:", mltq)
            likeDocs = searcher.search(mltq, 50).scoreDocs
            print("Like docs:")
            for likeDoc in likeDocs:
                doc = searcher.doc(likeDoc.doc)
                print('     title:', doc.get("title"), ", id:", likeDoc.doc)



if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    directory = MMapDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())
    analyzer = EnglishAnalyzer()
    analyzer = ShingleAnalyzerWrapper(analyzer, 3)
    run(searcher, analyzer, reader)
    del searcher
