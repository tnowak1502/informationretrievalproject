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

def evaluateParams(searcher, analyzer, reader, params, noToRetrieve=20, n=5, prints=False):
    """
    Evaluates customMoreLikeThis using the ground truth and the given parameters.

    Parameters
    ----------
    searcher: A lucene searcher over the index
    analyzer: The analyzer to parse queries and to pass to customMoreLikeThis. Should be the same as the analyzer used to build the index.
    reader: A lucene reading over the index
    params: A dict containing the parameters for customMoreLikeThis (maxLen, minTf, minDocFreq, noOfTerms)
    noToRetrieve: The amount of relevant documents to retrieve based on the moreLikeThis query
    n: Where to measure precision aside from the overall precision
    prints: Flag to indicate if the results for individual ground truth entries should be displayed

    Returns
    ----------
    evaluator.finalEval(): A dict containing the average precision, precision@N, recall, and f1 score over the entire ground truth
    """

    #initiate evaluator and extract parameters
    evaluator = Evaluator(groundtruth, n)
    maxLen = params["maxLen"]
    minTf = params["minTf"]
    minDocFreq = params["minDocFreq"]
    noOfTerms = params["noOfTerms"]

    #iterate over grundtruth
    for game in groundtruth.keys():
        if prints:
            print(game)

        #find game in index
        query = QueryParser("title", analyzer).parse(prune_unwanted(game))
        scoreDocs = searcher.search(query, 10).scoreDocs

        found = False
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)

            if prune_unwanted(doc.get("title")) == prune_unwanted(game):
                found = True

                #generate moreLikeThisQuery using the given parameters
                mltq = customMoreLikeThis(scoreDoc.doc, analyzer, reader, maxLen, minTf, minDocFreq, noOfTerms)
                likeDocs = searcher.search(mltq, noToRetrieve).scoreDocs
                retrieved = []

                for likeDoc in likeDocs:
                    doc = searcher.doc(likeDoc.doc)
                    retrieved.append(doc.get("title"))

                #evaluate results
                eval = evaluator.evalSingle(game, retrieved)

                if prints:
                    print(eval)
                break

        if not found:
            if prints:
                print("Couldn't find", game)

    #compute overall evaluation
    return evaluator.finalEval()

def gridSearch(searcher, analyzer, reader, maxLens, minTfs, minDocFreqs, noOfTermss, noToRetrieve=20, n=5, individualPrints=False):
    """
    Conducts a grid search for customMoreLikeThis over the given parameter values. For information on the parameters see customMoreLikeThis

    Parameters
    ----------
    searcher: A lucene searcher over the index
    analyzer: The analyzer to parse queries and to pass to customMoreLikeThis. Should be the same as the analyzer used to build the index.
    reader: A lucene reading over the index
    maxLens: An array containing the different values for maxLen to test
    minTfs: An array containing the different values for minTf to test
    minDocFreqs: An array containing the different values for minDocFreq to test
    noOfTermss: An array containing the different values for noOfTerms to test
    noToRetrieve: The amount of relevant documents to retrieve based on the moreLikeThis query
    n: Where to measure precision aside from the overall precision
    individualPrints: Flag to indicate if the results for individual ground truth entries should be displayed
    """

    #init dicts to store best results per metric
    best_prec = {"params": {}, "value": 0}
    best_rec = {"params": {}, "value": 0}
    best_f1 = {"params": {}, "value": 0}
    best_precN = {"params": {}, "value": 0}

    #iterate over parameters
    for maxLen in maxLens:
        for minTf in minTfs:
            for minDocFreq in minDocFreqs:
                for noOfTerms in noOfTermss:
                    params = {"maxLen": maxLen, "minTf": minTf, "minDocFreq": minDocFreq, "noOfTerms": noOfTerms}

                    #evaluate customMoreLikeThis with the given parameters
                    fullEval = evaluateParams(searcher, analyzer, reader, params, noToRetrieve=noToRetrieve, n=n, prints=individualPrints)
                    print()
                    print("FULL EVALUATION:", params)
                    print(fullEval)
                    print()

                    #update best metric results if necessary
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
    """
    Calculates the precision-recall curve for customMoreLikeThis with the given parameters and stores it in a .pkl file

    Parameters
    ----------
    searcher: A lucene searcher over the index
    analyzer: The analyzer to parse queries and to pass to customMoreLikeThis. Should be the same as the analyzer used to build the index.
    reader: A lucene reading over the index
    params: A dict containing the parameters for customMoreLikeThis (maxLen, minTf, minDocFreq, noOfTerms)
    """

    precisions = []
    recalls = []
    recall = 0
    noToRetrieve = 1
    #evaluate customMoreLikeThis while increasing the amount of retrieved documents
    while recall < 1:
        finalEval = evaluateParams(searcher, analyzer, reader, params, noToRetrieve=noToRetrieve)
        precision = finalEval["precision"]
        recall = finalEval["recall"]
        print(noToRetrieve, "precision:", precision, "| recall:", recall)
        if noToRetrieve > 2000:
            break
        noToRetrieve += 1
    #store results
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

    params = {"maxLen": 9, "minTf": 4, "minDocFreq": 2, "noOfTerms": 75}
    precisionRecallCurve(searcher, analyzer, reader, params)

    del searcher
