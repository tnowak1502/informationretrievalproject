#!/usr/bin/env python

INDEX_DIR = "index.index"

import sys, os, lucene, time, math

from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.queries.mlt import MoreLikeThis
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute, TermFrequencyAttribute

def mltByHand(docId, analyzer, reader, maxLen, minDocFreq, minIndFreq):
    start = time.time()

    doc = reader.storedFields().document(docId)
    contents = doc.get("content")
    ts = analyzer.tokenStream("content", contents)
    qt_count_doc = {}
    termAtt = ts.addAttribute(CharTermAttribute.class_)
    freqAtt = ts.addAttribute(TermFrequencyAttribute.class_)
    ts.reset()
    while ts.incrementToken():
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
    #print("FIRST PART:", end - start)
    start = time.time()
    query_tf_idf = {}
    maxCount = max(qt_count_doc.values())
    numDocs = reader.numDocs()
    for query_term, count in qt_count_doc.items():
        # print(query_term)
        try:
            query_tf_idf[query_term] = 0
            if len(query_term) < maxLen:
                if count >= minDocFreq:
                    term_freq = reader.docFreq(Term("content", query_term))#qt_count_index[query_term]
                    if term_freq >= minIndFreq:
                        query_tf_idf[query_term] = count * 0.5 /maxCount * math.log(
                            numDocs / term_freq)
        except KeyError:
            print("KeyError:", query_term)
            continue
    end = time.time()
    #print("SECOND PART:", end-start)
    start = time.time()
    sorted_query_terms = sorted(query_tf_idf.items(), key=lambda x: x[1], reverse=True)
    top_10_query_terms = sorted_query_terms[:20]
    res = ""
    for i in range(len(top_10_query_terms)):
        qt = top_10_query_terms[i][0]
        res += "content:" + qt + " "
    #print("res:", res)
    res = QueryParser("content", analyzer).parse(res)
    end = time.time()
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
        command = -1
        while True:
            command = input("Select: ")
            if int(command)-1 < 0 or int(command) > len(scoreDocs):
                print("Select a number within range")
            else:
                print()
                break
        mltq = mltByHand(scoreDocs[int(command)].doc, analyzer, reader, 9, 5, 4)
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
