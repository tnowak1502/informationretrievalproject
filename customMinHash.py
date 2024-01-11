import copy
import pickle

import pandas as pd
from nltk import ngrams
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from Evaluator import Evaluator

def shingle(text, k):
    shingles = ngrams(text, k, pad_right=True, right_pad_symbol="_")
    return list(shingles)

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
                for item in hashTable[hashIndex]:
                     candidatePair.append((key,item))
                hashTable[hashIndex].append(key)
            else:
                hashTable[hashIndex]=[key]
        print("test")
    return set(candidatePair)

def LSHSingle(file,bands,bucketSize,title):
    #file opendoen
    with open(file, 'rb') as file:
        signatures = pickle.load(file)
    bandSize=128/bands
    candidatePair=[]
    searchSig=signatures[title]
    keys = list(signatures.keys())
    for x in range(bands):
        signatureIndex=(int(bandSize*x),int(bandSize*(x+1)))
        bandedSearchSig=searchSig[signatureIndex[0]:signatureIndex[1]]
        sum = np.sum(bandedSearchSig)
        hashIndexSearchSig = lshHashFunc(sum,bucketSize)
        for key in keys:
            bandedSig=signatures[key][signatureIndex[0]:signatureIndex[1]]
            sum=np.sum(bandedSig)
            hashIndex=lshHashFunc(sum,bucketSize)
            if hashIndexSearchSig == hashIndex:
                candidatePair.append((title,key))
    setCandidate=set(candidatePair)
    setCandidate.remove((title,title))
    return setCandidate

def checkCandidates(candidatepairs,file):
    with open(file, 'rb') as file:
        signatures = pickle.load(file)
    scores=[]
    for item in candidatepairs:
        score = minhash_sim(signatures[item[0]],signatures[item[1]])
        if score == 0:
            continue
        pairScore=(item[0],item[1],score)
        scores.append(pairScore)
    return scores

def jaccCandidates(candidatepairs, k):
    with open("preprocessed_data", 'rb') as file:
        terms = pickle.load(file)
    scores=[]
    for item in candidatepairs:
        score = jaccard(terms[item[0]],terms[item[1]], k)
        if score == 0:
            continue
        pairScore=(item[0],item[1],score)
        scores.append(pairScore)
    return scores

def testEval(candidates):
    with open("groundtruth.json") as f_in:
        groundtruth = json.load(f_in)
    score=0
    for candidate in candidates:
        if candidate[0] in groundtruth:
            if candidate[1] in groundtruth[candidate[0]]:
                score+=1
    return score

def testEvalSingle(candidates,title):
    with open("groundtruth.json") as f_in:
        groundtruth = json.load(f_in)
    score = 0
    truth=groundtruth[title]
    for item in candidates:
        if item[1] in truth:
            score+=1
    return score/truth

def test_groundtruth(file, bands, bucketsize):
    with open("groundtruth.json") as f_in:
        groundtruth = json.load(f_in)
    evaluator = Evaluator(groundtruth, 5)
    for title in groundtruth:
        try:
            if "&" in title:
                title = title.replace("&", "&amp;")
            if title == "Zuma's Revenge":
                title = "Zuma's Revenge!"
            print(title)
            start = time.time()
            can = LSHSingle(file, bands, bucketsize, title)
            print(time.time()-start)
            print(len(can))
            scores = checkCandidates(can, file)
            top20 = sorted(scores, key=lambda x: x[2], reverse=True)[:20]
            retrieved = []
            for top in top20:
                retrieved.append(top[1])
            #print(retrieved)
            #print(groundtruth[title])
            if "&amp;" in title:
                title = title.replace("&amp;", "&")
            if title == "Zuma's Revenge!":
                title = "Zuma's Revenge"
            print(evaluator.evalSingle(title, retrieved))
        except Exception as e:
             print(e)
    print(evaluator.finalEval())

def createDataGraph(file):
    with open(file, 'rb') as file:
        signatures = pickle.load(file)
    #interValDict={"0-10":100,"10-20":80,"20-30":60,"30-40":40,"40-50":51,"50-60":22,"60-70":64,"70-80":20,"80-90":18,"90-100":3}
    interValDict={"0-10":0,"10-20":0,"20-30":0,"30-40":0,"40-50":0,"50-60":0,"60-70":0,"70-80":0,"80-90":0,"90-100":0}
    signatures2=copy.deepcopy(signatures)
    for x in signatures:
        for y in signatures2:
            if x !=y:
                score=minhash_sim(signatures[x],signatures[y])
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
    intervals = list(interValDict.keys())
    values = list(interValDict.values())
    plt.bar(intervals, values, color='blue',
            width=0.8)
    plt.xlabel("similarity betweendocuments(in %)")
    plt.ylabel("Number of documents")
    plt.title("Number of documents in funciton of their similarity with eachother")
    plt.show()
    plt.bar(intervals, values, color='blue',
            width=0.8)
    plt.yscale("log")
    plt.ylabel("Number of documents(logscale)")
    plt.title("Number of documents in funciton of their similarity with eachother(logscale)")
    plt.show()

def demo(title):
    bands=[1,2,4,8,16]
    buckets=[10,100,1000,10000]
    print("-----")
    print("running demo for: "+title+" with Sha1")
    print("-----")
    for band in bands:
        for bucket in buckets:
            can= LSHSingle("sha1_minhash_index", band, bucket,title)
            print(str(band)+" bands and "+str(bucket)+" buckets gives "+str(len(can))+"candidate pairs")
        print("-----Change bands-----")

    print("-----")
    print("running demo for: "+title+" with mmh3")
    print("-----")
    for band in bands:
        for bucket in buckets:
            can= LSHSingle("mmh3_minhash_index", band, bucket,title)
            print(str(band)+" bands and "+str(bucket)+" buckets gives "+str(len(can))+"candidate pairs")
        print("-----Change bands-----")


if __name__ == "__main__":
    demo("Batman: The Telltale Series")
