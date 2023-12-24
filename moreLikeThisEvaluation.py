import sys, os, lucene, json, pickle

from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.index import DirectoryReader, IndexReader, Term
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import MMapDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity

from utils import prune_unwanted
from customMoreLikeThis import customMoreLikeThis
from Evaluator import Evaluator


groundtruth = json.load(open("groundtruth.json","r"))

def evaluateWithParams(searcher, analyzer, reader, params, noToRetrieve=20, n=5, prints=False):
    evaluator = Evaluator(groundtruth, n)
    maxLen = params["maxLen"]
    minTf = params["minTf"]
    minDocFreq = params["minDocFreq"]
    noOfTerms = params["noOfTerms"]

    for game in groundtruth.keys():
        if prints:
            print(game)
        query = QueryParser("title", analyzer).parse(prune_unwanted(game))
        scoreDocs = searcher.search(query, 10).scoreDocs

        found = False
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)

            if prune_unwanted(doc.get("title")) == prune_unwanted(game):
                found = True

                mltq = customMoreLikeThis(scoreDoc.doc, analyzer, reader, maxLen, minTf, minDocFreq, noOfTerms)
                likeDocs = searcher.search(mltq, noToRetrieve).scoreDocs
                retrieved = []

                for likeDoc in likeDocs:
                    doc = searcher.doc(likeDoc.doc)
                    retrieved.append(doc.get("title"))

                eval = evaluator.evalSingle(game, retrieved)

                if prints:
                    print(eval)
                break

        if not found:
            if prints:
                print("Couldn't find", game)

    return evaluator.finalEval()

def gridSearch(searcher, analyzer, reader, maxLens, minTfs, minDocFreqs, noOfTermss, noToRetrieve=20, n=5, individualPrints=False):
    best_prec = {"params": {}, "value": 0}
    best_rec = {"params": {}, "value": 0}
    best_f1 = {"params": {}, "value": 0}
    best_precN = {"params": {}, "value": 0}

    for maxLen in maxLens:
        for minTf in minTfs:
            for minDocFreq in minDocFreqs:
                for noOfTerms in noOfTermss:
                    params = {"maxLen": maxLen, "minTf": minTf, "minDocFreq": minDocFreq, "noOfTerms": noOfTerms}

                    fullEval = evaluateWithParams(searcher, analyzer, reader, params, noToRetrieve=noToRetrieve, n=n, prints=individualPrints)
                    print()
                    print("FULL EVALUATION:", params)
                    print(fullEval)
                    print()

                    prec = fullEval["precision"]
                    rec = fullEval["recall"]
                    precN = fullEval["precision@"+str(n)]
                    f1 = fullEval["f1"]

                    if prec > best_prec["value"]:
                        best_prec["value"] = prec
                        best_prec["params"] = params
                    if rec > best_rec["value"]:
                        best_rec["value"] = rec
                        best_rec["params"] = params
                    if f1 > best_f1["value"]:
                        best_f1["value"] = f1
                        best_f1["params"] = params
                    if precN > best_precN["value"]:
                        best_precN["value"] = precN
                        best_precN["params"] = params

    print("BEST VALUES AND PARAMS PER METRIC:")
    print("PRECISION:  ", best_prec)
    print("RECALL:     ", best_rec)
    print("F1:         ", best_f1)
    print("PRECISION@"+str(n)+":", best_precN)

def precisionRecallCurve(searcher, analyzer, reader, params):
    precisions = []
    recalls = []
    recall = 0
    noToRetrieve = 1
    while recall < 1:
        finalEval = evaluateWithParams(searcher, analyzer, reader, params, noToRetrieve=noToRetrieve)
        precision = finalEval["precision"]
        recall = finalEval["recall"]
        print(noToRetrieve, "precision:", precision, "| recall:", recall)
        if noToRetrieve > 2000:
            break
        noToRetrieve += 1
    pickle.dump((precisions, recalls), open("precisionRecallCurve.pkl", "wb"))

if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print('lucene', lucene.VERSION)
    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    directory = MMapDirectory(Paths.get(os.path.join(base_dir, "index.index")))

    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())
    analyzer = EnglishAnalyzer()

    #gridSearch(searcher, analyzer, reader, maxLens=[9], minTfs=[4], minDocFreqs=[2], noOfTermss=[75], n=5, noToRetrieve=20, individualPrints=False)

    params = params = {"maxLen": 9, "minTf": 4, "minDocFreq": 2, "noOfTerms": 75}
    precisionRecallCurve(searcher, analyzer, reader, params)

    del searcher
