def prune_unwanted(document):
    """
    Prunes unwanted characters from given string 'document'. 
    This can be done after reading the dataset from the csv 
    file.

    Parameters
    ----------
    document : str : input string we want to prune of unwanted characters

    Returns
    -------
    document : str : string after replacing unwanted characters.
    """
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