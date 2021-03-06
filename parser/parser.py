from pickle import FALSE, TRUE
import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | NP VP Conj VP
NP -> N | NP P NP | Det NP | AP NP | NP AP
VP -> V | VP NP | VP Det NP | VP P NP | AP VP | VP AP
AP -> Adj | Adv | Adj AP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # Tokenize lowercased sentence
    tokens = nltk.word_tokenize(sentence.lower())
    words = []

    for token in tokens:
        valid = False
        # If token contains an alphabet then keep it
        for letter in token:
            if letter.isalpha():
                valid = True
        if valid:
            words.append(token)
    
    return words

def contains_np(subtree):
    # If subtree is NP return true
    if subtree.label() == "NP":
        return True

    # If subtree only has one child, return false
    # Since no node has NP as a single child
    if len(subtree) == 1:
        return False
    
    # Recursivly find any NP futher down in subtrees
    for subsubtree in subtree:
        if contains_np(subsubtree):
            return True
    
    return False

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    nps = []
    possible_labels = ("NP", "VP", "S") # Only these nodes can contain NP

    for subtree in tree:
        # Skip if no NP inside
        node = subtree.label()
        if not contains_np(subtree):
            continue
        
        # Recursively find NP
        if tree.label() in possible_labels:
            subsubtree = np_chunk(subtree)
            for np in subsubtree:
                nps.append(np)
    
    # Append terminal node if it satisfies criterias
    if tree.label() == "NP" and not contains_np(subtree):
        nps.append(tree)
    
    return nps


if __name__ == "__main__":
    main()
