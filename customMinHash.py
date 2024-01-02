import copy
import pickle

import pandas
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import json
from utils import prune_unwanted
import time

#stop_words = set(stopwords.words('english'))

def shingle(text, k):
    shingles = ngrams(text, k, pad_right=True, right_pad_symbol="_")
    return list(shingles)

# def stemming_and_stopword_removal(text):
#     stemmer = PorterStemmer()
#     terms = [stemmer.stem(term) for term in text.lower().split() if term not in stop_words]
#     return terms

# def preprocess(text, k):
#     terms = stemming_and_stopword_removal(text)
#     return shingle(terms, k)

def hash_func(seed):
    def hash_func(x):
        return int(hashlib.sha1(str(x).encode('utf-8') + str(seed).encode('utf-8')).hexdigest(), 16)
    return hash_func

def create_hash_funcs(seed, num_hashes):
    hash_funcs = []
    for i in range(num_hashes):
        hash_funcs.append(hash_func(seed+i))
    return hash_funcs

def create_signature(hash_funcs, preprocessed):
    hash_values = np.full((len(hash_funcs), len(preprocessed)), np.inf)
    for i, func in enumerate(hash_funcs):
        hash_values[i] = list(map(func, preprocessed))
    minhash_values = np.min(hash_values, axis=1)
    return minhash_values

def minhash(terms, k, hash_funcs):
    shingles = shingle(terms, k)
    return create_signature(hash_funcs, shingles)

def check_candidate(sig1, sig2):
    num_equal = np.sum(sig1 == sig2)
    if num_equal > 0:
        return True
    return False

def minhash_sim(sig1, sig2):
    num_equal = np.sum(sig1 == sig2)
    return num_equal/len(sig1)

def jaccard(terms1, terms2, k):
    preprocessed1 = set(shingle(terms1, k))
    preprocessed2 = set(shingle(terms2, k))
    intersection = preprocessed1.intersection(preprocessed2)
    union = preprocessed1.union(preprocessed2)
    return float(len(intersection))/float(len(union))
def jaccardSignature(x,y):
    return len(x.intersection(y))/len(x.union(y))
def lshHashFunc(val,bucketSize):
    return val%bucketSize
def LSH(file,bands,bucketSize):
    #file opendoen
    with open(file, 'rb') as file:
        signatures = pickle.load(file)
    bandSize=128/bands
    candidatePair=[]
    for x in range(bands):
        hashTable={}
        signatureIndex=(int(bandSize*x),int(bandSize*(x+1)))
        for key in signatures:
            bandedSig=signatures[key][signatureIndex[0]:signatureIndex[1]]
            sum=np.sum(bandedSig)
            hashIndex=lshHashFunc(sum,bucketSize)
            if hashIndex in hashTable:
                candidatePair.append((key,hashTable[hashIndex]))
            else:
                hashTable[hashIndex]=key
    return candidatePair

def testEval(candidates,size):
    with open("groundtruth.json") as f_in:
        groundtruth = json.load(f_in)
    score=0
    for candidate in candidates:
        if candidate[0] in groundtruth:
            if candidate[1] in groundtruth[candidate[0]]:
                score+=1
    return score
def createDataGraph(file):
    with open(file, 'rb') as file:
        signatures = pickle.load(file)
    counter=0
    #interValDict={"0-10":100,"10-20":80,"20-30":60,"30-40":40,"40-50":51,"50-60":22,"60-70":64,"70-80":20,"80-90":18,"90-100":3}
    interValDict={"0-10":0,"10-20":0,"20-30":0,"30-40":0,"40-50":0,"50-60":0,"60-70":0,"70-80":0,"80-90":0,"90-100":0}
    signatures2=copy.deepcopy(signatures)
    for x in signatures:
        for y in signatures2:
            if x !=y:
                score=minhash_sim(signatures[x],signatures[y])
                print(score)
                if 0 <= score < 0.1:
                    interValDict["0-10"] += 1
                elif 0.1 <= score < 0.2:
                    interValDict["10-20"] += 1
                elif 0.2 <= score < 0.3:
                    interValDict["20-30"] += 1
                elif 0.3 <= score < 0.4:
                    interValDict["30-40"] += 1
                elif 0.4 <= score < 0.5:
                    interValDict["40-50"] += 1
                elif 0.5 <= score < 0.6:
                    interValDict["50-60"] += 1
                elif 0.6 <= score < 0.7:
                    interValDict["60-70"] += 1
                elif 0.7 <= score < 0.8:
                    interValDict["70-80"] += 1
                elif 0.8 <= score < 0.9:
                    interValDict["80-90"] += 1
                elif 0.9 <= score <= 1.0:
                    interValDict["90-100"] += 1
        del signatures2[x]
    print(interValDict)
    intervals = list(interValDict.keys())
    values = list(interValDict.values())
    plt.bar(intervals, values, color='blue',
            width=0.8)
    plt.xlabel("similarity betweendocuments(in %)")
    plt.ylabel("Number of documents")
    plt.title("Number of documents in funciton of their similarity with eachother")
    plt.show()
if __name__ == "__main__":
    # createDataGraph("minhash_index")
    can = LSH("minhash_index",1,1)
    print(len(can))
    print(testEval(can,10))


