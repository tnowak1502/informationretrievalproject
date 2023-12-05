from org.apache.lucene.analysis.core import LowerCaseFilter, WhitespaceTokenizer
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.minhash import MinHashFilter
from org.apache.lucene.analysis.shingle import ShingleFilter

class MinHashAnalyzer(PythonAnalyzer):
    def __init__(self):
        PythonAnalyzer.__init__(self)

    def createComponents(self, fieldName):
        print("Creating components!!!!!!")
        source = WhitespaceTokenizer()
        result = LowerCaseFilter(source)
        result = ShingleFilter(source, 5)
        result = MinHashFilter(result, MinHashFilter.DEFAULT_HASH_COUNT, MinHashFilter.DEFAULT_BUCKET_COUNT, MinHashFilter.DEFAULT_HASH_SET_SIZE, False)
        return Analyzer.TokenStreamComponents(source, result)