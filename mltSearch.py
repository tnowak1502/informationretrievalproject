#!/usr/bin/env python

INDEX_DIR = "index.index"

import sys, os, lucene, time, math

from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
from org.apache.lucene.index import DirectoryReader, Term, SingleTermsEnum, SlowImpactsEnum
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute, TermFrequencyAttribute

def mltByHand(docId, analyzer, reader, maxLen, minTf, minDocFreq):
    """
    Extracts the terms with the highest tf-id-score from a given document and turns them into a query to be used for
    relevant document retrieval.

    Parameters
    ----------
    docId : The lucene index id of the document to extract terms from.
    analyzer : The analyzer that is used to turn the document's contents into a token stream, this should be the same as the analyzer used to build the index.
    reader : The reader reading the lucene index to search in.
    maxLen : tuning parameter to exclude terms that go over a certain length (to avoid terms that are extremely specific)
    minTf : The minimum amount of times a term needs to appear in the document to be taken into account during score calculation
    minDocFreq : The minimum amount of documents in the index a term needs to appear in to be taken into account during score calculation

    Returns
    ----------
    res : A query consisting of a string concatenating the top extracted query terms together.
    """

    #start = time.time()

    #retrieve the document and its contents and turn them into a token stream for analysis
    doc = reader.storedFields().document(docId)
    contents = doc.get("content")
    ts = analyzer.tokenStream("content", contents)
    #collect the term frequency for each term in the document
    qt_tf = {}
    termAtt = ts.addAttribute(CharTermAttribute.class_)
    freqAtt = ts.addAttribute(TermFrequencyAttribute.class_)
    ts.reset()
    while ts.incrementToken():
        term = termAtt.toString()
        #filter stopwords and terms that are too long
        if term in EnglishAnalyzer.ENGLISH_STOP_WORDS_SET or len(term) > maxLen:
            continue
        freq = freqAtt.getTermFrequency()
        if qt_tf.get(term) is None:
            qt_tf[term] = freq
        else:
            qt_tf[term] += freq
    ts.close()

    #end = time.time()
    #print("FIRST PART:", end - start)
    #start = time.time()

    #calculate tf-idf-scores for each term
    query_tf_idf = {}
    #maxCount = max(qt_tf.values())
    numDocs = reader.numDocs()
    for query_term, tf in qt_tf.items():
        if tf >= minTf:
            doc_freq = reader.docFreq(Term("content", term))
            if doc_freq >= minDocFreq:
                idf = math.log(1+ (numDocs - doc_freq+0.5)/(doc_freq + 0.5))
                query_tf_idf[query_term] = tf*idf
    #end = time.time()
    #print("SECOND PART:", end-start)
    #start = time.time()

    #retrieve the top 30 query terms and turn them into a concatenated query
    sorted_query_terms = sorted(query_tf_idf.items(), key=lambda x: x[1], reverse=True)
    top_query_terms = sorted_query_terms[:30]
    res = ""
    for i in range(len(top_query_terms)):
        qt = top_query_terms[i][0]
        res += qt + " "
    #print("res:", res)
    res = QueryParser("content", analyzer).parse(res)
    #end = time.time()
    #print("THIRD PART:", end - start)
    return res

def run(searcher, analyzer, reader):
    while True:
        print()
        print("Hit enter with no input to quit.")
        command = input("Query: ")
        if command == '':
            return
        print()
        print("Searching for:", command)
        query = QueryParser("content", analyzer).parse(command)
        scoreDocs = searcher.search(query, 5).scoreDocs
        for i in range(len(scoreDocs)):
            scoreDoc = scoreDocs[i]
            doc = searcher.doc(scoreDoc.doc)
            print(i+1, 'title:', doc.get("title"), "| id:", scoreDoc.doc)
        print()
        print("Select document to find related documents for by typing its number")
        while True:
            command = input("Select: ")
            if int(command)-1 < 0 or int(command) > len(scoreDocs):
                print("Select a number within range")
            else:
                print()
                break
        mltq = mltByHand(scoreDocs[int(command)-1].doc, analyzer, reader, 9, 4, 2)
        #print("query:", mltq)
        likeDocs = searcher.search(mltq, 20).scoreDocs
        print("Relevant documents:")
        for likeDoc in likeDocs:
            doc = searcher.doc(likeDoc.doc)
            print('     title:', doc.get("title"), "| id:", likeDoc.doc)



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
