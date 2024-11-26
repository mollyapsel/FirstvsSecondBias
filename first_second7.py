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
  

    m=len(tokens)
    unique = typefr.keys()
    num = "WEAT7"
    # targets: math and arts
    # attributes: male and female
    weat7 = {'x':['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition'],
    'y': ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture'],
    'a': ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son'],
    'b': ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']}
    x, y, a, b, out = check_words(num, weat7['x'], weat7['y'], weat7['a'], weat7['b'], typefr)

    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(out + "\n")
    print(results)
    print(num + " complete") 