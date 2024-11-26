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
    num = "WEAT9"
    # targets: mental and physical disease
    # attributes: temporary and permanent
    weat9 = {'x':['sad', 'hopeless', 'gloomy', 'tearful', 'miserable', 'depressed'],
    'y': ['sick', 'illness', 'influenza', 'disease', 'virus', 'cancer'],
    'a':['impermanent', 'unstable', 'variable', 'fleeting', 'shortterm', 'brief', 'occasional'],
    'b': ['stable', 'always', 'constant', 'persistent', 'chronic', 'prolonged', 'forever']}
    x, y, a, b, out = check_words(num, weat9['x'], weat9['y'], weat9['a'], weat9['b'], typefr)

    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(out + "\n")
    print(results)
    print(num + " complete") 