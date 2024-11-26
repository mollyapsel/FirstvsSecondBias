import os 
import pickle 
from first_second_bias import *

nlp = spacy.load("en_core_web_sm")
STOP_WORDS = STOP_WORDS - set(['he', 'him', 'his', 'she','her','hers','always'])

def preprocess_wiki(examples):
    typefr=collections.Counter()
    all_processed_tokens = []
    #Tokenizing the sentence
    for example in examples:
        text = example.lower()
        processed_tokens = [token.lemma_ for token in nlp(text) if token.is_alpha and not token.is_punct and token.lemma_ not in STOP_WORDS]
        typefr.update(processed_tokens)
        all_processed_tokens.extend(processed_tokens)
    return all_processed_tokens, typefr

if __name__ == '__main__':

    with open("../wiki.txt", "r") as file:
        text_corpus = file.read().splitlines()
    file_path1 = 'preprocessed_tokens_wiki.pkl'
    file_path2 = 'typefr_wiki.pkl'

    tokens, typefr = preprocess_wiki(text_corpus)
    print("Done preprocessing")
    with open(file_path1, 'wb') as file:
        pickle.dump(tokens, file)  
    with open(file_path2, 'wb') as file:
        pickle.dump(typefr, file)
    m=len(tokens)
    unique = typefr.keys()
    print("Total words: ", m)
    print("Unique words: ", len(unique))