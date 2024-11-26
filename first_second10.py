import os 
import pickle 
from first_second_bias import *

if __name__ == '__main__':
    file_path1 = 'preprocessed_tokens_wiki.pkl'
    file_path2 = 'typefr_wiki.pkl'

    if os.path.exists(file_path1):
        # Load pre-processed tokens from the file
        with open(file_path1, 'rb') as file:
            tokens = pickle.load(file)
        with open(file_path2, 'rb') as file:
            typefr = pickle.load(file)        
    # else:
    #     # Process the dataset and save the tokens
    #     dataset = load_dataset("allenai/c4", "en.noblocklist", split = "validation", streaming=True)
    #     tokens, typefr = preprocess(dataset)
    #     with open(file_path1, 'wb') as file:
    #         pickle.dump(tokens, file)  
    #     with open(file_path2, 'wb') as file:
    #         pickle.dump(typefr, file)   

    m=len(tokens)
    unique = typefr.keys()
    num = "WEAT10"
    # targets: young and old people's names
    # attributes: pleasant and unpleasant
    weat10 = {'x':['Tiffany', 'Michelle', 'Cindy', 'Kristy', 'Brad', 'Eric', 'Joey', 'Billy'],
    'y': ['Ethel', 'Bernice', 'Gertrude', 'Agnes', 'Cecil', 'Wilbert', 'Mortimer', 'Edgar'],
    'a': ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy'],
    'b': ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']}
    x, y, a, b, out = check_words(num, weat10['x'], weat10['y'], weat10['a'], weat10['b'], typefr)

    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(out + "\n")
    print(results)
    print(num + " complete")   