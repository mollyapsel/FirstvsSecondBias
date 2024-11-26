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
    num = "WEAT4"
    # targets: European American and African American names
    # attributes: pleasant and unpleasant
    weat4 = {"x": "Brad, Brendan, Geoffrey, Greg, Brett, Jay, Matthew, Neil, Todd, Allison, Anne, Carrie, Emily, Jill, Laurie, Kristen, Meredith, Sarah".lower().split(", "), "y": "Darnell, Hakim, Jermaine, Kareem, Jamal, Leroy, Rasheed, Tremayne, Tyrone, Aisha, Ebony, Keisha, Kenya, Latonya, Lakisha, Latoya, Tamika, Tanisha".lower().split(", "), "a": ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation'], "b": ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy','bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer','evil', 'kill', 'rotten', 'vomit']}
    x, y, a, b, out = check_words(num, weat4['x'], weat4['y'], weat4['a'], weat4['b'], typefr)

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
    # firstandsecond_sims.to_csv('WEAT4_pair_sims_en.csv')