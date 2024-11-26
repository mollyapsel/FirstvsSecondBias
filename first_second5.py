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
    else:
        # Process the dataset and save the tokens
        dataset = load_dataset("allenai/c4", "en", split = "validation", streaming=True)
        tokens, typefr = preprocess(dataset)
        with open(file_path1, 'wb') as file:
            pickle.dump(tokens, file)  
        with open(file_path2, 'wb') as file:
            pickle.dump(typefr, file)
  

    m=len(tokens)
    unique = typefr.keys()
    print("Total words: ", m)
    print("Unique words: ", len(unique))
    
    num = "WEAT5"
    # targets: European American and African American names
    # attributes: pleasant and unpleasant
    weat5 = {"x": "Brad, Brendan, Geoffrey, Greg, Brett, Jay, Matthew, Neil, Todd, Allison, Anne, Carrie, Emily, Jill, Laurie, Kristen, Meredith, Sarah".lower().split(", "), "y": "Darnell, Hakim, Jermaine, Kareem, Jamal, Leroy, Rasheed, Tremayne, Tyrone, Aisha, Ebony, Keisha, Kenya, Latonya, Lakisha, Latoya, Tamika, Tanisha".lower().split(", "), "a": ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy'], "b":['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']}
    x, y, a, b, out = check_words(num, weat5['x'], weat5['y'], weat5['a'], weat5['b'], typefr)

    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(out + "\n")
    print(results)
    print(num + " complete")  
    # x_a = list(product(x, a))
    # x_b = list(product(x, b))
    # y_a = list(product(y, a))
    # y_b = list(product(y, b))
    # pairs = x_a + x_b + y_a + y_b
    # first_order = ppmi(pairs, tokens, m, typefr)
    # second_order = socpmi(pairs, tokens, 0.2, 1.5, m, typefr, unique)

    # first_order = {key: [value] for key, value in first_order.items()}
    # second_order = {key: [value] for key, value in second_order.items()}
    # f = pd.DataFrame.from_dict(first_order)
    # s = pd.DataFrame.from_dict(second_order)
    # firstandsecond_sims = pd.concat([f,s])
    # firstandsecond_sims.to_csv('WEAT5_pair_sims_nb.csv')