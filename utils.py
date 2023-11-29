# Prunes unwanted characters from given string 'document'. 
# This can be done after reading the dataset from the csv 
# file and merging Title and Sections.
# 
# Returns string 'document' after replacing unwanted characters.
def prune_unwanted(document):
    document = document.replace("[", "")
    document = document.replace("]", "")
    document = document.replace("\"", "")
    document = document.replace("\\n", "")
    document = document.replace("\\", "")
    document = document.replace("/", " ")
    document = document.replace(".", " ")
    document = document.replace(",", " ")
    document = document.replace("-", " ")
    document = document.replace("(", "")
    document = document.replace(")", "")
    document = document.replace(":", " ")
    document = document.replace(";", " ")
    document = document.replace("'s", "")

    return document