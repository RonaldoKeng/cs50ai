import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_map = dict()

    # Get os path to directory and get all file names
    os_path = os.path.join(os.getcwd(), directory)
    file_names = os.listdir(directory)

    # Map file content to file name
    for file_name in file_names:
        with open(os.path.join(os_path, file_name)) as f:
            file_map[f] = f.read()
    
    return file_map


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Get all tokens
    tokens = nltk.tokenize.word_tokenize(document.lower())
    final_tokens = []

    # Filter tokens with spec criterias
    for token in tokens:
        if token not in string.punctuation and token not in nltk.corpus.stopwords.words("english"):
            final_tokens.append(token)
    
    return final_tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_map = dict()
    unique_words = set(sum(documents.values(), []))

    # Calculate idf for each unique word
    for word in unique_words:
        document_count = 0
        for document in documents.values():
            if word in document:
                document_count += 1
        
        idf = math.log(len(documents) / document_count)
        idf_map[word] = idf
    
    return idf_map


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    query_tf_idf_map = dict()

    # For each query word in each file,
    # Calculate its tf-idf and sum it for each file
    for file in files:
        content = files[file]
        tf_idf_sum = 0
        for query_word in query:
            if query_word in content:
                tf_idf = idfs[query_word] * content.count(query_word)
                tf_idf_sum += tf_idf
        query_tf_idf_map[file] = tf_idf_sum
    
    top_files = [file for file, idf in sorted(query_tf_idf_map.items(), key=lambda item: item[1], reverse=True)]
    return top_files[:n]
            

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    importance_map = dict()

    # For each word in each sentence,
    # Calculate its idf and sum it for each sentence
    # Then calculate keyword density for each sentence
    for sentence in sentences:
        content = sentences[sentence]
        idf_sum = 0
        keyword_counts = 0
        
        for word in query:
            if word in content:
                idf_sum += idfs[word]
                keyword_counts += content.count(word)
        
        density = keyword_counts / len(content)
        importance_map[sentence] = (idf_sum, density)

    top_sentences = [sentence for sentence, metrics in sorted(importance_map.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)]
    return top_sentences[:n]

if __name__ == "__main__":
    main()
