import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import random

def read_text(file_name):
    sentences = []
    
    f_data = open(file_name, 'rb').read().decode(encoding='utf-8')
    f_data = [x for x in f_data if x != '\n']
    f_data = [x.replace('\n',' ') for x in f_data]
    f_data = ''.join(f_data) 
    text = f_data.split('. ') 
    
    for sentence in text:
        sentences.append(sentence.replace("^[a-zA-Z0-9!@#$&()-`+,/\"]", " ").split(" "))
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    # stop words are words like “a”, “the”, “is”, “are” and etc
    if stopwords is None: # if there are no stop words, then there are no stop words
        stopwords=[]
    
    # convert each sentence into a vector in order to compare it to other sentences (vectors) and see which ones are closest in similarity and convert it back to sentences
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1+sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1-cosine_distance(vector1,vector2)

def sim_matrix(sentences,stop_words):
    # Create an empty matrix of similar words
    similarity_matrix=np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: # if the sentence is the same go next
                continue
            similarity_matrix[idx1][idx2]=sentence_similarity(sentences[idx1],sentences[idx2], stop_words)
    
    return similarity_matrix

def generate_summary(fileN, leng=1):
    stop_words=stopwords.words('english')
    summarize_text=[]
        
    # split text
    sentences =  read_text(fileN)

    # generates the similarity matrix across sentences
    sentence_similarity_martix = sim_matrix(sentences, stop_words)  

    # ranks sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # sorts the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        
    #print("Indexes of top ranked_sentence are ", ranked_sentence)   
        
    for i in range(leng):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        
    return summarize_text
    #print("Summary: \n",". ".join(summarize_text))

generate_summary("essay.txt")
