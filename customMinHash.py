import pickle
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import hashlib
from utils import prune_unwanted
import time

stop_words = set(stopwords.words('english'))

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

if __name__ == "__main__":

    signatures = {}

    # Create hash functions that will be used by the minhash algorithm
    seed = 1
    num_hashes = 128
    hash_funcs = create_hash_funcs(seed, num_hashes)

    # length of the shingles
    k = 3

    # Reading all preprocessed documents from storage (This file can be generated with preprocess_dataset.py)
    documents = []
    with open("preprocessed_data", "rb") as ppd_file:
        documents = pickle.load(ppd_file)

    # Loop over documents to extract text terms and calculate minhash signature
    for title in documents.keys():

        terms = documents[title]

        sig = minhash(terms, k, hash_funcs)

        signatures[title] = sig

    # Store all minhash signatures as a pickled dictionary
    with open("minhash_index", "wb") as file:
        pickle.dump(signatures, file)
