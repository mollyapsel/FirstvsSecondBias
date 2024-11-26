#from first_second_bias import *
from mpi4py import MPI
from datasets import load_dataset
import itertools
#import os
import numpy as np
#import nltk
#nltk.download('stopwords')
#from nltk.stem import WordNetLemmatizer,PorterStemmer
#from nltk.tokenize import RegexpTokenizer
#from nltk.corpus import stopwords
import math
from collections import Counter
from tqdm import tqdm
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

#os.environ['CURL_CA_BUNDLE'] = ''
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Rank {rank} on node {MPI.Get_processor_name()} (out of {size} total ranks)")

nlp = spacy.load("en_core_web_sm")

# def process_text(text):
#     doc = nlp(text)
#     return [token.lemma_ for token in doc if token.is_alpha and not token.is_punct and token.lemma_ not in STOP_WORDS]

dataset = load_dataset("allenai/c4", "en", split="train")
token_frequencies = Counter()
all_processed_tokens = []
# Divide the workload among processes
for i, example in enumerate(dataset):
    if i % size == rank:
        text = example['text'].lower()
        processed_tokens = [token.lemma_ for token in nlp(text) if token.is_alpha and not token.is_punct and token.lemma_ not in STOP_WORDS]
        token_frequencies.update(processed_tokens)
        all_processed_tokens.extend(processed_tokens)

# Combine token frequencies from all processes
token_frequencies_global = comm.gather(token_frequencies, root=0)
# Gather processed tokens from all nodes
all_tokens_gathered = comm.gather(all_processed_tokens, root=0)

# On the root process, merge token frequencies from all processes
if rank == 0:
    print("Finished counting")
    typefr = Counter()
    for freq_counter in token_frequencies_global:
        typefr.update(freq_counter)

    combined_tokens = []
    for tokens_list in all_tokens_gathered:
        combined_tokens.extend(tokens_list)

    related = [('bank', 'money'), ('leash', 'dog'), ('pilot', 'airplane'), ('vase', 'flower'), ('honey', 'bee'), ('cow', 'milk'), ('garage', 'car'), ('postman', 'mail'), ('hammer', 'nail'), ('beak', 'bird'), ('camel', 'desert'), ('scissors', 'paper'), ('chef', 'food'), ('crown', 'king'), ('sheep', 'wool'), ('milk', 'cow'), ('mechanic', 'car'), ('pumpkin', 'halloween'), ('drugstore', 'medicine'), ('circus', 'clown'), ('beach', 'sand'), ('candle', 'fire'), ('bee', 'honey'), ('spider', 'web'), ('boat', 'water'), ('lamp', 'light'), ('king', 'queen'), ('restaurant', 'food'), ('dragon', 'fire'), ('key', 'lock'), ('plate', 'food'), ('foot', 'shoe'), ('monkey', 'banana'), ('key', 'door'), ('santa', 'christmas'), ('mouse', 'cheese'), ('river', 'water'), ('axe', 'tree'), ('desert', 'sand'), ('axe', 'wood'), ('man', 'woman'), ('bone', 'dog'), ('hospital', 'doctor'), ('foot', 'toe'), ('witch', 'broom'), ('king', 'crown'), ('castle', 'king'), ('bed', 'pillow'), ('shoe', 'lace'), ('baker', 'bread'), ('gift', 'christmas'), ('woman', 'man'), ('pencil', 'eraser'), ('waitress', 'restaurant'), ('ship', 'ocean'), ('vase', 'water'), ('bed', 'sleep'), ('santa', 'present'), ('hospital', 'patient'), ('table', 'chair'), ('boy', 'girl'), ('hand', 'finger'), ('waitress', 'food'), ('rabbit', 'carrot'), ('student', 'school'), ('camel', 'hump'), ('crown', 'queen'), ('circus', 'elephant'), ('postman', 'mailbox'), ('tree', 'leaf'), ('coffee', 'cream'), ('candle', 'wax'), ('river', 'fish'), ('child', 'toy'), ('church', 'god'), ('pencil', 'paper'), ('barn', 'horse'), ('cheese', 'mouse'), ('knife', 'fork'), ('computer', 'keyboard'), ('girl', 'boy'), ('barn', 'cow'), ('ship', 'water'), ('cake', 'icing'), ('candy', 'child'), ('builder', 'house'), ('teacher', 'school'), ('lion', 'mane'), ('plate', 'fork'), ('cage', 'bird'), ('cake', 'birthday'), ('school', 'teacher'), ('pie', 'apple'), ('airplane', 'pilot'), ('hospital', 'nurse'), ('bee', 'hive'), ('chicken', 'egg'), ('builder', 'construction'), ('school', 'book')]
    similar = [('vanish', 'disappear'), ('quick', 'rapid'), ('creator', 'maker'), ('stupid', 'dumb'), ('insane', 'crazy'), ('happy', 'cheerful'), ('large', 'big'), ('cow', 'cattle'), ('area', 'region'), ('large', 'huge'), ('simple', 'easy'), ('bizarre', 'strange'), ('student', 'pupil'), ('attorney', 'lawyer'), ('occur', 'happen'), ('hallway', 'corridor'), ('teacher', 'instructor'), ('inform', 'notify'), ('smart', 'intelligent'), ('weird', 'odd'), ('taxi', 'cab'), ('drizzle', 'rain'), ('happy', 'glad'), ('scarce', 'rare'), ('protect', 'defend'), ('declare', 'announce'), ('boundary', 'border'), ('plead', 'beg'), ('adversary', 'opponent'), ('cop', 'sheriff'), ('business', 'company'), ('pact', 'agreement'), ('corporation', 'business'), ('strange', 'odd'), ('victory', 'triumph'), ('essential', 'necessary'), ('abundance', 'plenty'), ('weird', 'strange'), ('journey', 'trip'), ('physician', 'doctor'), ('decide', 'choose'), ('movie', 'film'), ('task', 'job'), ('roam', 'wander'), ('shore', 'coast'), ('contemplate', 'think'), ('crucial', 'important'), ('acquire', 'get'), ('friend', 'buddy'), ('heroine', 'hero'), ('danger', 'threat'), ('hard', 'difficult'), ('fast', 'rapid'), ('inspect', 'examine'), ('champion', 'winner'), ('locate', 'find'), ('attention', 'awareness'), ('anger', 'fury'), ('create', 'make'), ('inexpensive', 'cheap'), ('assignment', 'task'), ('motor', 'engine'), ('delightful', 'wonderful'), ('alcohol', 'gin'), ('buddy', 'companion'), ('wonderful', 'terrific'), ('acquire', 'obtain'), ('achieve', 'accomplish'), ('bubble', 'suds'), ('crib', 'cradle'), ('style', 'fashion'), ('noticeable', 'obvious'), ('explore', 'discover'), ('aggression', 'hostility'), ('create', 'build'), ('pretend', 'imagine'), ('apparent', 'obvious'), ('sheep', 'lamb'), ('bad', 'awful'), ('certain', 'sure'), ('horse', 'mare'), ('beach', 'seashore'), ('purse', 'bag'), ('make', 'construct'), ('clarify', 'explain'), ('area', 'zone'), ('couple', 'pair'), ('arrange', 'organize'), ('orthodontist', 'dentist'), ('keep', 'posse'), ('confident', 'sure'), ('sea', 'ocean'), ('expand', 'grow'), ('dreary', 'dull'), ('think', 'rationalize'), ('begin', 'originate'), ('proclaim', 'announce'), ('harsh', 'cruel'), ('aviation', 'flight'), ('motel', 'inn')]

    #tokens = clean(dataset)

    m=len(combined_tokens)
    unique = vocab_size
    #m_prime = sum([typefr[t]**0.75 for t in unique])

    delta_vals = [0.5, 0.7, 1, 3, 5, 7, 10]
    gamma_vals = [1.5, 2, 2.5, 3, 3.5, 4]
    reldf = pd.DataFrame(index=delta_vals, columns=gamma_vals)
    simdf = pd.DataFrame(index=delta_vals, columns=gamma_vals)

    def socpmi(word_pairs, processed_tokens, dvals, gvals, df, context_window_size = 5):
        # second order co-occurrence PMI
        #socpmi_scores = {}
        
        a = context_window_size
        

        for words in tqdm(word_pairs):
            w1 = words[0]
            w2 = words[1]
            neighboursw1=Counter()
            n2w1=[]
            neighboursw2=Counter()
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
            #pmiw1={}
            pmi1_old = {}
            for t in neighboursw1.keys():
                #pmiw1[t]= max(0, math.log(neighboursw1[t]* m_prime/((typefr[t]**0.75)*typefr[w1]),2))
                pmi1_old[t] = max(0, math.log(neighboursw1[t]* m/(typefr[t]*typefr[w1]),2))

            #print(neighboursw2)
            #pmiw2={}
            pmi2_old = {}
            for t in neighboursw2.keys():
                #pmiw2[t]= max(0,math.log((neighboursw2[t]*m_prime/((typefr[t]**0.75)*typefr[w2])),2))
                pmi2_old[t] = max(0, math.log(neighboursw2[t]* m/(typefr[t]*typefr[w2]),2))


            pmiw1_sorted = sorted(pmiw1_old, key=pmiw1_old.get, reverse=True)
            pmiw2_sorted = sorted(pmiw2_old, key=pmiw2_old.get, reverse=True)
            
            # for i in range(20):
            #     print(w1, pmiw1_sorted[i], pmiw1[pmiw1_sorted[i]], pmi1_old[pmiw1_sorted[i]])
            
            # for i in range(20):
            #     print(w2, pmiw2_sorted[i], pmiw2[pmiw2_sorted[i]], pmi2_old[pmiw2_sorted[i]])
            for delta in dvals:
            
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
                for gamma in gvals:
                    for i in range(0,b1):
                        for j in range(0,b2):
                            if pmiw1_sorted[i]==pmiw2_sorted[j]:
                                #print(pmiw1_sorted[i])
                                betasumw1+=math.pow(pmiw2[pmiw1_sorted[i]],gamma)
                                betasumw2+=math.pow(pmiw1[pmiw1_sorted[i]],gamma)

                    #print("betasumw1:", betasum
                    #print("betasumw2:", betasumw2)
                    if b1==0:
                        b1 = 1
                    if b2==0:
                        b2 = 1
                    similarity= betasumw1/b1 + betasumw2/b2
                    
                    df.at[delta, gamma] += similarity

        return None

    socpmi(related, combined_tokens, delta_vals, gamma_vals, reldf)
    socpmi(similar, combined_tokens, delta_vals, gamma_vals, simdf)

    reldf = reldf/len(related)
    simdf = simdf/len(similar)

    reldf.to_csv('relatedc4test.csv')
    simdf.to_csv('similarc4test.csv')

    

# # Create a mapping from tokens to indices
# token_to_index = {token: idx for idx, (token, _) in enumerate(typefr.most_common())}

# # Save token frequencies to a memory-mapped file
# vocab_size = len(token_to_index)
# freq_array = np.memmap('token_frequencies.dat', dtype='int32', mode='w+', shape=(vocab_size,))

# # Populate the memory-mapped array with token frequencies
# for token, freq in typefr.items():
#     freq_array[token_to_index[token]] = freq




# #typefr=collections.Counter()
# def clean(examples):
# 	stat=[]
# 	stop_words=set(stopwords.words("english"))

# 	#Tokenizing the sentence
# 	tokenizer= RegexpTokenizer(r'\w+')
# 	tokenized = [tokenizer.tokenize(ex["text"].lower()) for ex in examples]

# 	words = list(itertools.chain.from_iterable(tokenized))

# 	#Stemmer and Lemmatizer instance created
# 	ps=PorterStemmer()
# 	lemmatizer= WordNetLemmatizer()
# 	pronouns = ['he', 'him', 'his', 'she','her','hers']
# 	#Lemmatizing words and adding to the final array if they are not stopwords
# 	for w in words:
# 		if w not in stop_words or w in pronouns:
# 			w=lemmatizer.lemmatize(w)
# 			stat.append(w)
# 			typefr[w] += 1

# 	return stat


