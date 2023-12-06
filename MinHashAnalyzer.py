from org.apache.lucene.analysis.core import LowerCaseFilter
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.analysis import Analyzer, StopFilter
from org.apache.lucene.analysis.minhash import MinHashFilter
from org.apache.lucene.analysis.shingle import ShingleFilter
from org.apache.lucene.analysis.en import EnglishPossessiveFilter, EnglishAnalyzer, PorterStemFilter

class MinHashAnalyzer(PythonAnalyzer):
    def __init__(self):
        PythonAnalyzer.__init__(self)

    def createComponents(self, fieldName):
        source = StandardTokenizer()
        result = LowerCaseFilter(source)
        result = ShingleFilter(result, 5)
        result = MinHashFilter(result, 4, 512, 8, True)
        return Analyzer.TokenStreamComponents(source, result)