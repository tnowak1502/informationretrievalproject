import lucene

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from java.nio.file import Paths


def create_index(documents, index_path):
    """
    Creates a Lucene index or adds to an existing Lucene index at specified path

    Parameters
    ----------
    documents : list<dict> : The collection of documents to add to the Lucene index. This collection is structured as a list of dictionaries. 
    Each dictionary has 2 keys ('title' and 'content').
    index_path : str : The specified path where to Lucene index will be stored.

    """
    analyzer = EnglishAnalyzer()
    index_config = IndexWriterConfig(analyzer)
    index_directory = FSDirectory.open(Paths.get(index_path))

    # Create an IndexWriter
    writer = IndexWriter(index_directory, index_config)

    for doc_id, document in enumerate(documents):
        add_document(writer, doc_id, document)

    # Close the IndexWriter to finalize the index
    writer.close()

def add_document(writer, doc_id, document):
    """
    Adds the specified document with unique identifier 'doc_id' to the given Lucene index 'writer'.

    Parameters
    ----------
    writer : IndexWriter : The Lucene IndexWriter object that will add the document to the Lucene index.
    doc_id : int : A unique identifier for the document. In this implementation 'doc_id' is an integer.
    document: dict : The document that is added to the Lucene index. This document is a dictionary with 2 keys ('title' and 'content')
    """
    # Initialize new Document object
    doc = Document()

    # Create fieldtype that is used for each of the fields in the Document object.
    # The parameters stored and tokenized are set to True.
    # The field's value string will now be stored and tokenized in the index.
    fieldtype = FieldType()
    fieldtype.setStored(True)
    fieldtype.setTokenized(True)

    # Add a field for the document ID
    doc.add(Field("doc_id", str(doc_id), fieldtype))

    # Add a field for the document title
    doc.add(Field("title", document["title"], fieldtype))

    # Add a field for the document content
    doc.add(Field("content", document["content"], fieldtype))

    # Add the document to the index
    writer.addDocument(doc)

if __name__ == "__main__":
    lucene.initVM()
    print(lucene.VERSION)

    import pandas as pd
    from utils import prune_unwanted

    # Read 'video_games.txt' as a CSV file
    reader = pd.read_csv("video_games.txt", sep=",")

    documents = []
    
    # Loop over entries in CSV output to build list of dictionaries for the dataset
    for i in range(len(reader)):
        document = dict()
        document["title"] = reader["Title"][i]
        document["content"] = prune_unwanted(reader["Sections"][i])
        documents.append(document)

    # Add given collection of 'documents' to Lucene index at './index'
    create_index(documents, "index")