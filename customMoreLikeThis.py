from org.apache.lucene.analysis.tokenattributes import CharTermAttribute, TermFrequencyAttribute, TypeAttribute, PayloadAttribute, BytesTermAttribute
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import Term
from org.apache.lucene.search import BooleanQuery, BooleanClause, TermQuery
import math


def customMoreLikeThis(docId, analyzer, reader, maxLen, minTf, minDocFreq, noOfTerms):
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
    noOfTerms : The number of query terms to extract

    Returns
    ----------
    res : A query consisting of the top extracted query terms.
    """

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

    #calculate tf-idf-scores for each term
    query_tf_idf = {}
    numDocs = reader.numDocs()
    for query_term, tf in qt_tf.items():
        if tf >= minTf:
            doc_freq = reader.docFreq(Term("content", term))
            if doc_freq >= minDocFreq:
                idf = math.log(1 + (numDocs - doc_freq + 0.5)/(doc_freq + 0.5))
                query_tf_idf[query_term] = tf*idf

    #retrieve the top query terms and turn them into a concatenated query
    sorted_query_terms = sorted(query_tf_idf.items(), key=lambda x: x[1], reverse=True)
    top_query_terms = sorted_query_terms[:noOfTerms]
    builder = BooleanQuery.Builder()
    for i in range(len(top_query_terms)):
        qt = top_query_terms[i][0]
        builder.add(TermQuery(Term("content", qt)), BooleanClause.Occur.SHOULD)
    res = builder.build()
    return res