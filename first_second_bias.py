import itertools
import nltk
import numpy as np
import collections
from tqdm import tqdm
import math
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from itertools import product
from itertools import combinations
from datasets import load_dataset
import random
import operator


# def preprocess(examples):
#     typefr=collections.Counter()
#     stat=[]
#     stop_words=set(stopwords.words("english"))
# 	#Tokenizing the sentence
#     tokenizer= RegexpTokenizer(r'\w+')
#     tokenized = [tokenizer.tokenize(t) for t in examples["text"]]
#     words = list(itertools.chain.from_iterable(tokenized))

#     #Stemmer and Lemmatizer instance created
#     ps=PorterStemmer()
#     lemmatizer= WordNetLemmatizer()
#     pronouns = ['he', 'him', 'his', 'she','her','hers']
#     #Lemmatizing words and adding to the final array if they are not stopwords
#     for w in words:
# 	    if w not in stop_words or w in pronouns:
# 		    w = lemmatizer.lemmatize(w)
#           stat.append(w)
#           typefr[w] += 1
# 	return stat, typefr


def ppmi(word_pairs, tokens, m, typefr, context_window_size = 5):
    
    # Initialize a dictionary to store co-occurrence counts for your word pairs
    co_occurrence_counts = collections.Counter()

    # Iterate through the tokens and count co-occurrences of your word pairs within the context window
    for i in tqdm(range(m)):
        for words in word_pairs:
            if words[0] == tokens[i]:
                for j in range(max(0, i - context_window_size), min(len(tokens), i + context_window_size + 1)):
                    if words[1] == tokens[j]:
                        co_occurrence_counts[(words[0], words[1])] += 1
        
    # Calculate the PMI for each word pair
    ppmi_scores = {}
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
        return ppmi_scores

def socpmi(word_pairs, tokens, delta, gamma, m_prime, typefr, unique, context_window_size = 5):
    # second order co-occurrence PMI
    socpmi_scores = {}
    
    a = context_window_size
    

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
                for j in range(1,a+1):
                    neighboursw1[tokens[i+j]]+=1
                    neighboursw1[tokens[i-j]]+=1

            elif w2==tokens[i]:
                curr_window = tokens[max(0, i - a):min(len(tokens), i + a + 1)]
                if w1 in curr_window:
                    continue
                #neighboursw2[tokens[i]]+=1
                for j in range(1,a+1):
                    neighboursw2[tokens[i+j]]+=1
                    neighboursw2[tokens[i-j]]+=1   

        #print(neighboursw1)
        pmiw1={}
        #pmi1_old = {}
        for t in neighboursw1.keys():
            pmiw1[t]= max(0, math.log(neighboursw1[t]* m_prime/((typefr[t]**0.75)*typefr[w1]),2))
            #pmi1_old[t] = max(0, math.log(neighboursw1[t]* m/(typefr[t]*typefr[w1]),2))

        #print(neighboursw2)
        pmiw2={}
        #pmi2_old = {}
        for t in neighboursw2.keys():
            pmiw2[t]= max(0,math.log((neighboursw2[t]*m_prime/((typefr[t]**0.75)*typefr[w2])),2))
            #pmi2_old[t] = max(0, math.log(neighboursw2[t]* m/(typefr[t]*typefr[w2]),2))


        pmiw1_sorted = sorted(pmiw1, key=pmiw1.get, reverse=True)
        pmiw2_sorted = sorted(pmiw2, key=pmiw2.get, reverse=True)
        
        # for i in range(20):
        #     print(w1, pmiw1_sorted[i], pmiw1[pmiw1_sorted[i]], pmi1_old[pmiw1_sorted[i]])
        
        # for i in range(20):
        #     print(w2, pmiw2_sorted[i], pmiw2[pmiw2_sorted[i]], pmi2_old[pmiw2_sorted[i]])
        
        b1= math.floor((math.pow(math.log10(typefr[w1]),2)* math.log(len(unique),2))/delta)
        b2= math.floor((math.pow(math.log10(typefr[w2]),2) * math.log(len(unique),2))/delta)

        # print("b1:", b1)
        # print("b2:", b2)        
        # print(pmiw1_sorted)
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

        #print("betasumw1:", betasumw1)
        #print("betasumw2:", betasumw2)
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
  p = np.sum(stats > obs) / len(stats)
  return p

def check_words(num, X,Y,A,B, typefr):
    # check if missing words in X or Y or A or B; if yes, remove that word and a word from the counterpart set
    xrem = set()
    yrem = set()
    arem = set()
    brem = set()
    for word in X:
        if word not in typefr.keys():
            xrem.add(word)
    for word in Y:
        if word not in typefr.keys():
            yrem.add(word)
    for word in A:
        if word not in typefr.keys():
            arem.add(word)
    for word in B:
        if word not in typefr.keys():
            brem.add(word)
    x1 = set(X) - xrem
    y1 = set(Y) - yrem
    a1 = set(A) - arem
    b1 = set(B) - brem

    if len(x1) > len(y1):
        diff = len(x1) - len(y1)
        extra = sorted(X, key= operator.itemgetter(typefr))[:diff]
        x1 = x1 - set(extra)
    elif len(y1) > len(x1):
        diff = len(y1) - len(x1)
        extra = sorted(Y, key= operator.itemgetter(typefr))[:diff]
        y1 = y1 - set(extra)
    if len(a1) > len(b1):
        diff = len(a1) - len(b1)
        extra = sorted(A, key= operator.itemgetter(typefr))[:diff]
        a1 = a1 - set(extra)
    elif len(b1) > len(a1):
        diff = len(b1) - len(a1)
        extra = sorted(B, key= operator.itemgetter(typefr))[:diff]
        b1 = b1 - set(extra)    
    
    out = num + ": \n X: " + x1 + "\n Y: " + y1 + "\n A: " + a1 + "\n B: " + b1
    return list(x1), list(y1), list(a1), list(b1), out

#def run_tests(x,y,a,b,tokens)
if __name__ == '__main__':
    dataset = load_dataset("allenai/c4", "en.noblocklist", split = "train", streaming=True)
    tokens, typefr = preprocess(dataset)
    lr= WordNetLemmatizer()
    m=len(tokens)
    unique = typefr.keys()
    m_prime = sum([typefr[t]**0.75 for t in unique])

    df = {'test': ['WEAT1','WEAT2','WEAT3','WEAT4','WEAT5','WEAT6','WEAT7','WEAT8','WEAT9','WEAT10'], 'first_order_es': [], 'second_order_es': [], 'ratio_es':[]}
    outstr = ""

    weat1 = {'x': ['aster', 'clover', 'hyacinth', 'marigold', 'poppy', 'azalea', 'crocus', 'iris', 'orchid', 'rose', 'bluebell', 'daffodil', 'lilac', 'pansy', 'tulip', 'buttercup', 'daisy', 'lily', 'peony', 'violet', 'carnation', 'gladiola','magnolia', 'petunia', 'zinnia'],
    'y': ['ant', 'caterpillar', 'flea', 'locust', 'spider', 'bedbug', 'centipede', 'fly', 'maggot', 'tarantula', 'bee', 'cockroach', 'gnat', 'mosquito', 'termite', 'beetle', 'cricket', 'hornet', 'moth', 'wasp', 'blackfly', 'dragonfly', 'horsefly', 'roach', 'weevil'],
    'a': ['caress', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend', 'heaven', 'loyal', 'pleasure', 'diamond', 'gentle', 'honest', 'lucky', 'rainbow', 'diploma', 'gift', 'honor', 'miracle', 'sunrise', 'family', 'happy', 'laughter', 'paradise', 'vacation'],
    'b': ['abuse', 'crash', 'filth', 'murder', 'sickness', 'accident', 'death', 'grief', 'poison', 'stink', 'assault', 'disaster', 'hatred', 'pollute', 'tragedy', 'divorce', 'jail', 'poverty', 'ugly', 'cancer', 'kill', 'rotten', 'vomit', 'agony', 'prison']}
    x, y, a, b, out = check_words("WEAT1", weat1['x'], weat1['y'], weat1['a'], weat1['b'], typefr)
    outstr += out
    run_tests


    # Create a new text file
    with open("new_file.txt", "w") as f:
        # Write the string to the file
        f.write("This is a new text file.")