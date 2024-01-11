import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity

from customMoreLikeThis import customMoreLikeThis
from utils import prune_unwanted

#Based on SearchFiles.py from the the pylucene samples folder

def run(searcher, analyzer, reader):
    """
    Prompts the user with an open search and then allows them to select a document from the results for which relevant documents are retrieved.

    Parameters
    ----------
    searcher : IndexSearcher : A lucene searcher over the index
    analyzer : Analyzer : The analyzer to parse queries and to pass to customMoreLikeThis. Should be the same as the analyzer used to build the index.
    reader : DirectoryReader : A lucene reader reading the index
    """
    while True:
        print()
        print("Hit enter with no input to quit.")
        command = input("Query: ")
        if command == '':
            return
        print()
        print("Searching for:", command)
        query = QueryParser("title", analyzer).parse(prune_unwanted(command))
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
                continue
            else:
                print()
                break
        mltq = customMoreLikeThis(scoreDocs[int(command) - 1].doc, analyzer, reader, 9, 4, 2, 60)
        likeDocs = searcher.search(mltq, 21).scoreDocs
        print("Relevant documents:")
        # remove the first result, because it will be the game itself
        for likeDoc in likeDocs[1:]:
            doc = searcher.doc(likeDoc.doc)
            print('     title:', doc.get("title"), "| id:", likeDoc.doc)



if __name__ == '__main__':
    #start lucene and locate index
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    directory = MMapDirectory(Paths.get(os.path.join(base_dir, "index.index")))

    #load index, initialize searcher and analyzer
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())
    analyzer = EnglishAnalyzer()

    run(searcher, analyzer, reader)
    del searcher
