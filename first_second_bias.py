import itertools
import nltk
import numpy as np
import collections
from tqdm import tqdm
import math
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from itertools import product
from itertools import combinations
from datasets import load_dataset
import random
import operator
import os 
import pickle 
import pandas as pd

nlp = spacy.load("en_core_web_sm")
STOP_WORDS = STOP_WORDS - set(['he', 'him', 'his', 'she','her','hers','always'])

def preprocess_half(docs, format):
    typefr1=collections.Counter()
    typefr2=collections.Counter()
    tokens1 = []
    tokens2 = []
    grp = 0
    for doc in docs:
        if format == "hf":
            doc = doc['text']
        text = doc.lower()
        processed_tokens = [token.lemma_ for token in nlp(text) if token.is_alpha and not token.is_punct and token.lemma_ not in STOP_WORDS]
        if grp == 0:
            typefr1.update(processed_tokens)
            tokens1.extend(processed_tokens)
            grp += 1
        else:
            typefr2.update(processed_tokens)
            tokens2.extend(processed_tokens)
            grp -= 1  
    return typefr1, tokens1, typefr2, tokens2         
        


def preprocess(examples):
    typefr=collections.Counter()
    all_processed_tokens = []
    #Tokenizing the sentence
    for example in examples:
        text = example['text'].lower()
        processed_tokens = [token.lemma_ for token in nlp(text) if token.is_alpha and not token.is_punct and token.lemma_ not in STOP_WORDS]
        typefr.update(processed_tokens)
        all_processed_tokens.extend(processed_tokens)
    return all_processed_tokens, typefr


def ppmi(word_pairs, tokens, m, typefr, context_window_size = 5):
    
    # Initialize a dictionary to store co-occurrence counts for your word pairs
    co_occurrence_counts = collections.Counter()

    # Iterate through the tokens and count co-occurrences of your word pairs within the context window
    for i in range(m):
        for words in word_pairs:
            if words[0] == tokens[i]:
                for j in range(max(0, i - context_window_size), min(m, i + context_window_size + 1)):
                    if words[1] == tokens[j]:
                        co_occurrence_counts[(words[0], words[1])] += 1
        
    # Calculate the PMI for each word pair
    ppmi_scores = {}
    raw_freqs = {}
    for words in word_pairs:
        word1 = words[0]
        word2 = words[1]

        # if p_word1 == 0:
        #     return word1
        # if p_word2 == 0:
        #     return word2
        if co_occurrence_counts[(word1, word2)] == 0:
            ppmi = 0
            #print(word1, word2)
        else:
            ppmi = max(0, math.log(co_occurrence_counts[(word1, word2)]* m/(typefr[word1]*typefr[word2]),2))
        ppmi_scores[(word1, word2)] = ppmi
    return ppmi_scores, co_occurrence_counts # return sim. scores, co-occurrence freq. for each word pair, 

def socpmi(word_pairs, tokens, delta, gamma, m, typefr, unique, context_window_size = 5):
    # second order co-occurrence PMI
    socpmi_scores = {}
    
    a = context_window_size
    m_prime = sum([typefr[t]**0.75 for t in unique])

    for words in tqdm(word_pairs):
        w1 = words[0]
        w2 = words[1]
        neighboursw1=collections.Counter()
        n2w1=[]
        neighboursw2=collections.Counter()
        n2w2=[]
        for i in range(len(tokens)):
            if w1==tokens[i]:
                curr_window = tokens[max(0, i - a):min(len(tokens), i + a + 1)]
                if w2 in curr_window:
                    continue # skip if w2 is in the context window of w1

                #neighboursw1[tokens[i]]+=1
                curr_window.remove(w1)
                neighboursw1.update(curr_window)

            elif w2==tokens[i]:
                curr_window = tokens[max(0, i - a):min(len(tokens), i + a + 1)]
                if w1 in curr_window:
                    continue
                #neighboursw2[tokens[i]]+=1
                curr_window.remove(w2)
                neighboursw2.update(curr_window)

        #print(neighboursw1)
        pmiw1={}
        #pmiw1_old = {}
        for t in neighboursw1.keys():
            pmiw1[t]= max(0, math.log(neighboursw1[t]* m_prime/((typefr[t]**0.75)*typefr[w1]),2))
            #pmiw1_old[t] = max(0, math.log(neighboursw1[t]* m/(typefr[t]*typefr[w1]),2))

        #print(neighboursw2)
        pmiw2={}
        #pmiw2_old = {}
        for t in neighboursw2.keys():
            pmiw2[t]= max(0,math.log((neighboursw2[t]*m_prime/((typefr[t]**0.75)*typefr[w2])),2))
            #pmiw2_old[t] = max(0, math.log(neighboursw2[t]* m/(typefr[t]*typefr[w2]),2))


        pmiw1_sorted = sorted(pmiw1, key=pmiw1.get, reverse=True)
        pmiw2_sorted = sorted(pmiw2, key=pmiw2.get, reverse=True)
        
        # for i in range(20):
        #     print(w1, pmiw1_sorted[i], pmiw1[pmiw1_sorted[i]], pmiw1_old[pmiw1_sorted[i]])
        
        # for i in range(20):
        #     print(w2, pmiw2_sorted[i], pmiw2[pmiw2_sorted[i]], pmiw2_old[pmiw2_sorted[i]])
        
        b1= math.floor((math.pow(math.log10(typefr[w1]),2)* math.log(len(unique),2))/delta)
        b2= math.floor((math.pow(math.log10(typefr[w2]),2) * math.log(len(unique),2))/delta)

        # print("b1:", b1)
        # print("b2:", b2)        
        # print(pmiw1_sorted)
        # print(pmiw2_sorted)
        if b1>len(pmiw1_sorted):
            b1=len(pmiw1_sorted)
        
        if b2>len(pmiw2_sorted):
            b2=len(pmiw2_sorted)

        # print("b1:", b1)
        # print("b2:", b2)

        betasumw1=0
        betasumw2=0

        # print("pmiw1_sorted:", pmiw1_sorted[:b1])
        # print("pmiw2_sorted:", pmiw2_sorted[:b2])
        
        for i in range(0,b1):
            for j in range(0,b2):
                if pmiw1_sorted[i]==pmiw2_sorted[j]:
                    #print(pmiw1_sorted[i])
                    betasumw1+=math.pow(pmiw2[pmiw1_sorted[i]],gamma)
                    betasumw2+=math.pow(pmiw1[pmiw1_sorted[i]],gamma)

        # print("betasumw1:", betasumw1)
        # print("betasumw2:", betasumw2)
        if b1==0:
          b1 = 1
        if b2==0:
          b2 = 1
        similarity= betasumw1/b1 + betasumw2/b2
        
        socpmi_scores[(w1, w2)] = similarity

    return socpmi_scores

def effect_size(X, Y, A, B, sim_scores):
    # for each x in set X, s= average similarity of x to each word a in set A - average similarity of x to each word b in set B
    # for each y in set Y, s= average similarity of y to each word a in set A - average similarity of y to each word b in set B
    # effect size = average of s for each x in X - average of s for each y in Y divided by the standard deviation of s for each x in X and y in Y
    
    #TODO: save all of the s_x and s_y and get a t-test
    s_x = []
    s_y = []
    for x in X:
        s = np.mean([sim_scores[(x, a)] for a in A]) - np.mean([sim_scores[(x, b)] for b in B])
        s_x.append(s)

    for y in Y:
        s = np.mean([sim_scores[(y, a)] for a in A]) - np.mean([sim_scores[(y, b)] for b in B])
        s_y.append(s)

    diff = np.mean(s_x) - np.mean(s_y)
    pooled_sd = np.std(s_x+s_y)
    return diff / pooled_sd

def get_parts(X, Y, fewer=False):
  if len(X)!=len(Y):
    return "uneven target set lengths"

  if fewer:
    all_combinations = [random.sample(X+Y, len(X)) for _ in range(1000)]
  else:
    all_combinations = combinations(X+Y, len(X))

  equal_splits = []
  for combo in all_combinations:
      remaining = list(X+Y)
      for item in combo:
          remaining.remove(item)
      equal_splits.append((list(combo), remaining))
  return equal_splits

def p_test(X, Y, A, B, sim_scores, equal_splits):
  s_w = {}
  for i in X+Y:
    s_w[i] = np.mean([sim_scores[(i, a)] for a in A]) - np.mean([sim_scores[(i, b)] for b in B])

  obs = sum([s_w[x] for x in X]) - sum([s_w[y] for y in Y])

  stats = np.array([sum([s_w[x] for x in part[0]]) - sum([s_w[y] for y in part[1]]) for part in equal_splits])
  numerator = np.sum(stats > obs)
  #print("Numerator: ", numerator)
  denominator = len(stats)
  #print("Denominator: ", denominator)
  p =  numerator / denominator 
  return p

def check_words(num, X,Y,A,B, typefr):
    # check if missing words in X or Y or A or B; if yes, remove that word and a word from the counterpart set
    xrem = set()
    yrem = set()
    arem = set()
    brem = set()
    X = [token.lemma_ for token in nlp(' '.join(X).lower())]
    Y = [token.lemma_ for token in nlp(' '.join(Y).lower())]
    A = [token.lemma_ for token in nlp(' '.join(A).lower())]
    B = [token.lemma_ for token in nlp(' '.join(B).lower())]

    for word in X:
        if word not in typefr.keys():
            xrem.add(word)
            print("OOV:", word, num)
    for word in Y:
        if word not in typefr.keys():
            yrem.add(word)
            print("OOV:", word, num)
    for word in A:
        if word not in typefr.keys():
            arem.add(word)
            print("OOV:", word, num)
    for word in B:
        if word not in typefr.keys():
            brem.add(word)
            print("OOV:", word, num)
    x1 = set(X) - xrem
    y1 = set(Y) - yrem
    a1 = set(A) - arem
    b1 = set(B) - brem

    if len(x1) > len(y1):
        diff = len(x1) - len(y1)
        extra = sorted(x1, key= lambda w: typefr[w])[:diff]
        x1 = x1 - set(extra)
    elif len(y1) > len(x1):
        #print("x:",len(x1))
        #print("y:",len(y1))
        diff = len(y1) - len(x1)
        extra = sorted(y1, key= lambda w: typefr[w])[:diff]
        # print(extra)
        # print(len(y1))
        y1 = y1 - set(extra)
        #print(len(y1))
    if len(a1) > len(b1):
        diff = len(a1) - len(b1)
        extra = sorted(a1, key= lambda w: typefr[w])[:diff]
        a1 = a1 - set(extra)
    elif len(b1) > len(a1):
        diff = len(b1) - len(a1)
        extra = sorted(b1, key= lambda w: typefr[w])[:diff]
        b1 = b1 - set(extra)    
    
    out = num + ": \n X: " + str(x1) + "\n Y: " + str(y1) + "\n A: " + str(a1) + "\n B: " + str(b1) +"\n"
    return list(x1), list(y1), list(a1), list(b1), out

def run_tests(x, y, a, b, tokens, m, typefr, unique, fewer=False):
    pairs = list(product(x+y, a+b))
    first_order = ppmi(pairs, tokens, m, typefr)
    # delta and gamma found after testing with corpus
    #second_order = socpmi(pairs, tokens, 0.4, 1.75, m, typefr, unique) # used for TASA
    second_order = socpmi(pairs, tokens, 0.2, 1.5, m, typefr, unique) # used for c4 and wiki tests
    parts = get_parts(x, y, fewer)
    es1 = effect_size(x, y, a, b, first_order)
    p1 = p_test(x, y, a, b, first_order, parts)
    es2 = effect_size(x, y, a, b, second_order)
    p2 = p_test(x, y, a, b, second_order, parts)
    return es1, p1, es2, p2

if __name__ == '__main__':
    file_path1 = 'preprocessed_tokens.pkl'
    file_path2 = 'typefr.pkl'  
    # with open("tasaSentDocs.txt", "r") as file:
    #     text_corpus = file.read() 
    #dataset = load_dataset("allenai/c4", "en", split = "validation", streaming=True)
    #paragraphs = [paragraph.strip().replace('\n', '') for paragraph in text_corpus.split('\n \n')]
    tokens, typefr = preprocess(dataset)
    # with open(file_path1, 'wb') as file:
    #     pickle.dump(tokens, file)  
    # with open(file_path2, 'wb') as file:
    #     pickle.dump(typefr, file)
 
    print("Done preprocessing")
    m=len(tokens)
    unique = typefr.keys()
    print("Total words: ", m)
    print("Unique words: ", len(unique))

    df = {'test': ['WEAT1','WEAT2','WEAT3','WEAT4','WEAT5','WEAT6','WEAT7','WEAT8','WEAT9','WEAT10'], 'first_order_es': [], 'first_order_p':[], 'second_order_es': [], 'second_order_p':[]}
    outstr = ""

    num = "WEAT1"
    # targets: flowers and insects
    # attributes: pleasant and unpleasant
    weat1 = {'x': ['aster', 'clover', 'hyacinth', 'marigold', 'poppy', 'azalea', 'crocus', 'iris', 'orchid', 'rose', 'bluebell', 'daffodil', 'lilac', 'pansy', 'tulip', 'buttercup', 'daisy', 'lily', 'peony', 'violet', 'carnation', 'gladiola','magnolia', 'petunia', 'zinnia'],
    'y': ['ant', 'caterpillar', 'flea', 'locust', 'spider', 'bedbug', 'centipede', 'fly', 'maggot', 'tarantula', 'bee', 'cockroach', 'gnat', 'mosquito', 'termite', 'beetle', 'cricket', 'hornet', 'moth', 'wasp', 'blackfly', 'dragonfly', 'horsefly', 'roach', 'weevil'],
    'a': ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation'],
    'b': ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'kill', 'rotten', 'vomit', 'agony', 'prison']}
    x, y, a, b, out = check_words(num, weat1['x'], weat1['y'], weat1['a'], weat1['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")

    num = "WEAT2"
    # targets: instruments and weapons
    # attributes: pleasant and unpleasant
    weat2 = {'x': ['bagpipe', 'cello', 'guitar', 'lute', 'trombone', 'banjo', 'clarinet', 'harmonica', 'mandolin', 'trumpet', 'bassoon', 'drum', 'harp', 'oboe', 'tuba', 'bell', 'fiddle', 'harpsichord', 'piano', 'viola', 'bongo', 'flute', 'horn', 'saxophone', 'violin'],
    'y': ['arrow', 'club', 'gun', 'missile', 'spear', 'axe', 'dagger', 'harpoon', 'pistol', 'sword', 'blade', 'dynamite', 'hatchet', 'rifle', 'tank', 'bomb', 'firearm', 'knife', 'shotgun', 'teargas', 'cannon', 'grenade', 'mace', 'slingshot', 'whip'],
    'a': ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation'],
    'b': ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'kill', 'rotten', 'vomit', 'agony', 'prison']}
    x, y, a, b, out = check_words(num, weat2['x'], weat2['y'], weat2['a'], weat2['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")

    num = "WEAT3"
    # targets: European American and African American names
    # attributes: pleasant and unpleasant
    weat3 = {'x': ['Adam', 'Harry', 'Josh', 'Roger', 'Alan', 'Frank', 'Justin', 'Ryan', 'Andrew', 'Jack', 'Matthew', 'Stephen', 'Brad', 'Greg', 'Paul', 'Jonathan', 'Peter', 'Amanda', 'Courtney', 'Heather', 'Melanie', 'Katie', 'Betsy', 'Kristin', 'Nancy', 'Stephanie', 'Ellen', 'Lauren', 'Colleen', 'Emily', 'Megan', 'Rachel'],
    'y': ['Alonzo', 'Jamel', 'Theo', 'Alphonse', 'Jerome', 'Leroy', 'Torrance', 'Darnell', 'Lamar', 'Lionel', 'Tyree', 'Deion', 'Lamont', 'Malik', 'Terrence', 'Tyrone', 'Lavon', 'Marcellus', 'Wardell', 'Nichelle', 'Shereen', 'Ebony', 'Latisha', 'Shaniqua', 'Jasmine', 'Tanisha', 'Tia', 'Lakisha', 'Latoya', 'Yolanda', 'Malika', 'Yvette'],
    'a': ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation'],
    'b': ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy','bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer','evil', 'kill', 'rotten', 'vomit']}
    x, y, a, b, out = check_words(num, weat3['x'], weat3['y'], weat3['a'], weat3['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")
    
    num = "WEAT4"
    # targets: European American and African American names
    # attributes: pleasant and unpleasant
    weat4 = {'x': ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Meredith', 'Sarah'],
    'y': ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha'],
    'a': ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation'],
    'b': ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy','bomb', 'divorce', 'jail', 'poverty', 'ugly', 'cancer','evil', 'kill', 'rotten', 'vomit']}
    x, y, a, b, out = check_words(num, weat4['x'], weat4['y'], weat4['a'], weat4['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")

    num = "WEAT5"
    # targets: European American and African American names
    # attributes: pleasant and unpleasant
    weat5 = {'x': ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Matthew', 'Neil', 'Todd', 'Allison', 'Anne', 'Carrie', 'Emily', 'Jill', 'Laurie', 'Meredith', 'Sarah'],
    'y': ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tyrone', 'Aisha', 'Ebony', 'Keisha', 'Kenya', 'Lakisha', 'Latoya', 'Tamika', 'Tanisha'],
    'a': ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy'],
    'b': ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']}
    x, y, a, b, out = check_words(num, weat5['x'], weat5['y'], weat5['a'], weat5['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")

    num = "WEAT6"
    # targets: male and female names
    # attributes: career and family
    weat6 = {'x':['John', 'Paul', 'Mike', 'Kevin', 'Steve', 'Greg', 'Jeff', 'Bill'],
    'y': ['Amy', 'Joan', 'Lisa', 'Sarah', 'Diana', 'Kate', 'Ann', 'Donna'],
    'a': ['executive', 'management', 'professional', 'corporation', 'salary', 'office', 'business', 'career'],
    'b': ['home', 'parents', 'children', 'family', 'cousins', 'marriage', 'wedding', 'relatives']}
    x, y, a, b, out = check_words(num, weat6['x'], weat6['y'], weat6['a'], weat6['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")

    num = "WEAT7"
    # targets: math and arts
    # attributes: male and female
    weat7 = {'x':['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition'],
    'y': ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture'],
    'a': ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son'],
    'b': ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']}
    x, y, a, b, out = check_words(num, weat7['x'], weat7['y'], weat7['a'], weat7['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")

    num = "WEAT8"
    # targets: science and arts
    # attributes: male and female
    weat8 = {'x':['science', 'technology', 'physics', 'chemistry', 'Einstein', 'NASA', 'experiment', 'astronomy'],
    'y': ['poetry', 'art', 'Shakespeare', 'dance', 'literature', 'novel', 'symphony', 'drama'],
    'a':['brother', 'father', 'uncle', 'grandfather', 'son', 'he', 'his', 'him'],
    'b': ['sister', 'mother', 'aunt', 'grandmother', 'daughter', 'she', 'hers', 'her']}
    x, y, a, b, out = check_words(num, weat8['x'], weat8['y'], weat8['a'], weat8['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")

    num = "WEAT9"
    # targets: mental and physical disease
    # attributes: temporary and permanent
    weat9 = {'x':['sad', 'hopeless', 'gloomy', 'tearful', 'miserable', 'depressed'],
    'y': ['sick', 'illness', 'influenza', 'disease', 'virus', 'cancer'],
    'a':['impermanent', 'unstable', 'variable', 'fleeting', 'shortterm', 'brief', 'occasional'],
    'b': ['stable', 'always', 'constant', 'persistent', 'chronic', 'prolonged', 'forever']}
    x, y, a, b, out = check_words(num, weat9['x'], weat9['y'], weat9['a'], weat9['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")

    num = "WEAT10"
    # targets: young and old people's names
    # attributes: pleasant and unpleasant
    weat10 = {'x':['Tiffany', 'Michelle', 'Cindy', 'Kristy', 'Brad', 'Eric', 'Joey', 'Billy'],
    'y': ['Ethel', 'Bernice', 'Gertrude', 'Agnes', 'Cecil', 'Wilbert', 'Mortimer', 'Edgar'],
    'a': ['joy', 'love', 'peace', 'wonderful', 'pleasure', 'friend', 'laughter', 'happy'],
    'b': ['agony', 'terrible', 'horrible', 'nasty', 'evil', 'war', 'awful', 'failure']}
    x, y, a, b, out = check_words(num, weat10['x'], weat10['y'], weat10['a'], weat10['b'], typefr)
    outstr += out
    results = run_tests(x, y, a, b, tokens, m, typefr, unique, True)
    print(results)
    df['first_order_es'].append(results[0])
    df['first_order_p'].append(results[1])
    df['second_order_es'].append(results[2])
    df['second_order_p'].append(results[3])
    print(num + " complete")   

    pd.DataFrame.from_dict(df).to_csv('first_second_bias_c4withblock.csv')
    # Create a new text file
    with open("final_word_sets_withblock.txt", "w") as f:
        # Write the string to the file
        f.write(outstr)
    f.close()

    