import lucene

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from java.nio.file import Paths

def create_index(documents, index_path):
    analyzer = StandardAnalyzer()
    index_config = IndexWriterConfig(analyzer)
    index_directory = FSDirectory.open(Paths.get(index_path))

    # Create an IndexWriter
    writer = IndexWriter(index_directory, index_config)

    for doc_id, document in enumerate(documents):
        add_document(writer, doc_id, document)

    # Close the IndexWriter to finalize the index
    writer.close()

def add_document(writer, doc_id, document):
    doc = Document()

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

    reader = pd.read_csv("video_games.txt", sep=",")

    documents = []
    
    for i in range(len(reader)):
        document = dict()
        document["title"] = reader["Title"][i]
        document["content"] = prune_unwanted(reader["Sections"][i])
        documents.append(document)

    create_index(documents, "test_index")