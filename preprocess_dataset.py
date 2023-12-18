import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from utils import prune_unwanted

stop_words = set(stopwords.words('english'))

def stemming_and_stopword_removal(text):
    stemmer = PorterStemmer()
    terms = [stemmer.stem(term) for term in text.lower().split() if term not in stop_words]
    return terms

if __name__ == "__main__":
    # Dictionary for preprocessed documents
    preprocessed_documents = {}

    # Reading all documents from csv file
    documents = pd.read_csv("video_games.txt", sep=",").set_index("Title")

    # Loop over documents to extract and tokenize section text
    for title in documents.index:
        text = prune_unwanted(documents.at[title, "Sections"])
        terms = stemming_and_stopword_removal(text)
        
        preprocessed_documents[title] = terms
    
    # Store preprocessed text to pickle file
    with open("preprocessed_data", "wb") as ppd_file:
        pickle.dump(preprocessed_documents, ppd_file)