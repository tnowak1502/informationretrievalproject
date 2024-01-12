import pickle
from customMinHash import minhash
import mmh3
import hashlib

# ------------------------------------------------------
# Code to give a list of Hash functions using sha-1
# ------------------------------------------------------
def sha1_hash_func(seed):
    def hash_func(shingle):
        s = ""
        for el in shingle:
            s += el
        return int(hashlib.sha1(s.encode('utf-8') + str(seed).encode('utf-8')).hexdigest(), 16)
    return hash_func

def create_sha1_hash_funcs(seed, num_hashes):
    hash_funcs = []
    for i in range(num_hashes):
        hash_funcs.append(sha1_hash_func(seed+i))
    return hash_funcs

# ------------------------------------------------------------------
# Code to give a list of Hash functions using mmh3 (MurmurHash)
# ------------------------------------------------------------------
def mmh3_hash_func(seed):
    def hash_func(shingle):
        s = ""
        for el in shingle:
            s += el
        return mmh3.hash(s, seed, signed=False)
    return hash_func

def create_mmh3_hash_funcs(seed, num_hashes):
    hash_funcs = []
    for i in range(num_hashes):
        hash_funcs.append(mmh3_hash_func(seed+i))
    return hash_funcs

if __name__ == "__main__":
    ##GENERATE SIGNATURES##
    signatures = {}

    # Create hash functions that will be used by the minhash algorithm
    seed = 1
    num_hashes = 128
    hash_funcs = create_mmh3_hash_funcs(seed, num_hashes)

    # length of the shingles
    k = 3

    # Reading all preprocessed documents from storage (file can be generated with preprocess_dataset.py)
    documents = []
    with open("preprocessed_data", "rb") as ppd_file:
        documents = pickle.load(ppd_file)

    # Loop over documents to extract text terms and calculate minhash signature
    for title in documents.keys():

        terms = documents[title]

        sig = minhash(terms, k, hash_funcs)

        signatures[title] = sig

    # Store all minhash signatures as a pickled dictionary
    with open("mmh3_minhash_index128", "wb") as file:
        pickle.dump(signatures, file)