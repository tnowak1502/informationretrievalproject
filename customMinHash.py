import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import hashlib
from utils import prune_unwanted

stop_words = set(stopwords.words('english'))

def shingle(text, k):
    shingles = ngrams(text, k, pad_right=True, right_pad_symbol="_")
    return list(shingles)

def stemming_and_stopword_removal(text):
    stemmer = PorterStemmer()
    terms = [stemmer.stem(term) for term in text.lower().split() if term not in stop_words]
    return terms

def preprocess(text, k):
    terms = stemming_and_stopword_removal(text)
    return shingle(terms, k)

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
        for j, shingle in enumerate(preprocessed):
            hash_values[i, j] = func(shingle)
    minhash_values = np.min(hash_values, axis=1)
    return minhash_values

def minhash(text, k, seed, num_hashes):
    preprocessed = preprocess(text, k)
    hash_funcs = create_hash_funcs(seed, num_hashes)
    return create_signature(hash_funcs, preprocessed)

def check_candidate(sig1, sig2):
    num_equal = np.sum(sig1 == sig2)
    if num_equal > 0:
        return True
    return False

def minhash_sim(sig1, sig2):
    num_equal = np.sum(sig1 == sig2)
    return num_equal/len(sig1)

def jaccard(text1, text2, k):
    preprocessed1 = set(preprocess(text1, k))
    preprocessed2 = set(preprocess(text2, k))
    intersection = preprocessed1.intersection(preprocessed2)
    union = preprocessed1.union(preprocessed2)
    return float(len(intersection))/float(len(union))

if __name__ == "__main__":
    documents = pd.read_csv("video_games.txt", sep=",").set_index("Title")
    text1 = prune_unwanted(documents.at["Mafia II", "Sections"])
    text2 = prune_unwanted(documents.at["Mafia III", "Sections"])
    #text3 = prune_unwanted(documents.at["Grand Theft Auto V", "Sections"])
    #text4 = prune_unwanted(documents.at["Overwatch (video game)", "Sections"])
    seed = 1
    num_hashes = 128
    k = 3
    sig1 = minhash(text1, k, seed, num_hashes)
    sig2 = minhash(text2, k, seed, num_hashes)
    #sig3 = minhash(text3, k, seed, num_hashes)
    #sig4 = minhash(text4, k, seed, num_hashes)
    print(minhash_sim(sig1, sig2), jaccard(text1, text2, k))
    #print(check_candidate(sig1, sig3), jaccard(text1, text3, k))
    #print(check_candidate(sig1, sig4), jaccard(text1, text4, k))


