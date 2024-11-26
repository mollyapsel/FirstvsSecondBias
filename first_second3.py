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
    num = "WEAT3"
    # targets: European American and African American names
    # attributes: pleasant and unpleasant
    weat3 = {"x": "Adam, Chip, Harry, Josh, Roger, Alan, Frank, Ian, Justin, Ryan, Andrew, Fred, Jack, Matthew, Stephen, Brad, Greg, Jed, Paul, Todd, Brandon, Hank, Jonathan, Peter, Wilbur, Amanda, Courtney, Heather, Melanie, Sara, Amber, Crystal, Katie, Meredith, Shannon, Betsy, Donna, Kristin, Nancy, Stephanie, Bobbie-Sue, Ellen, Lauren, Peggy, Sue-Ellen, Colleen, Emily, Megan, Rachel, Wendy".lower().split(", "), "y": "Alonzo, Jamel, Lerone, Percell, Theo, Alphonse, Jerome, Leroy, Rasaan, Torrance, Darnell, Lamar, Lionel, Rashaun, Tyree, Deion, Lamont, Malik, Terrence, Tyrone, Everol, Lavon, Marcellus, Terryl, Wardell, Aiesha, Lashelle, Nichelle, Shereen, Temeka, Ebony, Latisha, Shaniqua, Tameisha, Teretha, Jasmine, Latonya, Shanise, Tanisha, Tia, Lakisha, Latoya, Sharise, Tashika, Yolanda, Lashandra, Malika, Shavonn, Tawanda, Yvette".lower().split(", "), "a": ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation'], "b": ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy','bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer','evil', 'kill', 'rotten', 'vomit']}
    x, y, a, b, out = check_words(num, weat3['x'], weat3['y'], weat3['a'], weat3['b'], typefr)
    
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(out + "\n")
    print(results)
    print(num + " complete")   

    # exact stimuli used in C4 en (with block list)
    # x = ['nancy', 'lauren', 'melanie', 'matthew', 'frank', 'andrew', 'jonathan', 'megan', 'harry', 'peter', 'heather', 'justin', 'greg', 'stephen', 'kristin', 'ellen', 'alan', 'jack', 'josh', 'katie', 'courtney', 'rachel', 'roger', 'brad', 'emily', 'paul', 'adam', 'ryan', 'stephanie', 'amanda', 'colleen']
    # y = ['alonzo', 'leroy', 'lakisha', 'theo', 'jasmine', 'darnell', 'malik', 'marcellus', 'lionel', 'jamel', 'deion', 'nichelle', 'tanisha', 'tyree', 'yolanda', 'jerome', 'tia', 'malika', 'latisha', 'wardell', 'latoya', 'lamar', 'ebony', 'terrence', 'torrance', 'alphonse', 'yvette', 'lavon', 'tyrone', 'lamont', 'shereen']
    # a = ['loyal', 'family', 'friend', 'miracle', 'cheer', 'lucky', 'caress', 'honest', 'honor', 'diploma', 'heaven', 'vacation', 'rainbow', 'diamond', 'health', 'laughter', 'pleasure', 'gentle', 'gift', 'happy', 'love', 'freedom', 'paradise', 'peace', 'sunrise']
    # b = ['disaster', 'death', 'vomit', 'stink', 'hatred', 'abuse', 'filth', 'poison', 'murder', 'poverty', 'pollute', 'sickness', 'assault', 'rotten', 'crash', 'jail', 'grief', 'cancer', 'bomb', 'divorce', 'kill', 'tragedy', 'ugly', 'evil', 'accident']
    
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
    # firstandsecond_sims.to_csv('WEAT3_pair_sims_en.csv')   