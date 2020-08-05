from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import pickle
import os

VOCAB_SIZE = 0
VERBOSE = False

def load_pickle_object(file):
    with open(file, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
	
def create_tde_objects(max_length = 64):
    
    processed_lines=[]

    for file in ['business', 'medical', 'personal', 'religious', 'research', 'sports', 'terror', 'political']:   
        fp=open('./dataset/dataset_'+file+'.txt', 'r')
        lines=fp.readlines()
        n=len(processed_lines)
        for line in lines:
            if len(line)>0:
                newString = line.lower()
                newString = re.sub(r"'s\b","",newString)
                newString = re.sub("[^a-zA-Z 0-9]", "", newString)
                if len(newString.strip()) > 0:
                    processed_lines.append(newString.strip())
        print(file, len(processed_lines)-n)
        fp.close()

    print("Number of sentences:", len(processed_lines))

    t = Tokenizer()
    t.fit_on_texts(processed_lines)
    vocab_size = len(t.word_index) + 1
    print("Vocab size=",vocab_size)

    with open('./tokenizer_small.pickle', 'wb') as handle:
        pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

    encoded_docs = t.texts_to_sequences(processed_lines)

    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='pre', truncating='pre')
    padded_docs = pad_sequences(padded_docs, maxlen=max_length+1, padding='post', truncating='pre')
    print(padded_docs.shape)
    words=t.word_docs
    
    fp=open('./sentences_small.txt', 'a+')
    fp.writelines(processed_lines)
    fp.close()
    
    with open('./padded_doc_small.pickle', 'wb') as handle:
        pickle.dump(padded_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    embeddings_index = dict()
    f = open('/home/sanket/nltk_data/glove.840B.300d.txt')
    for line in f:
        values = line.strip().lower().split()
        word = values[0]
        try:
            if word in words:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        except Exception as e:
            print(e)

    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    with open('./embedding_matrix_small.pickle', 'wb') as handle:
        pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Objects created!!!")

    
def create_file_structure(filename, domain='generic'):
    doc, vocab = get_padded_document(filename, max_length=64)
    path = './dataset/data/' + domain
    if not os.path.exists(path+'/train/'):
        print("Creating drirectory:", path+'/val/')
        os.makedirs(path+'/val/')
        os.makedirs(path+'/train/')
    for i, row in enumerate(doc):
        np.save(path+'/train/'+str(i), row)
    print("Creating validation set for", domain)
    for file in os.listdir(path+'/train/'):
        if '31' in file:
            os.rename(path+'/train/'+file, path+'/val/'+file)
    print("Done building file structure for:", domain, "!!")    
    
def create_dataset():
    
    path = './dataset/data/'
    for file in ['business', 'medical', 'personal', 'religious', 'research', 'sports', 'terror', 'political']:
        doc, vocab = get_padded_document('./dataset/dataset_'+file+'.txt', max_length=64)
        for i, row in enumerate(doc):
            np.save(path+file+'_'+str(i), row)
    print("Done building dataset!!")    
    
def create_all_files():
    for file in ['business', 'medical', 'personal', 'religious', 'research', 'sports', 'terror', 'political']:
        create_file_structure('./dataset/'+file+'.txt', domain=file)
    
def get_padded_document(filename, max_length=64): 
    
    processed_lines=[]
    fp=open(filename, 'r')
    lines=fp.readlines()
    n=len(processed_lines)
    for line in lines:
        if len(line)>0:
            newString = line.lower()
            newString = re.sub(r"'s\b","",newString)
            newString = re.sub("[^a-zA-Z 0-9]", "", newString)
            if len(newString.strip()) > 0:
                processed_lines.append(newString.strip())
    print("Number of sentences in", filename, "=",len(processed_lines)-n)
    fp.close()
    print(processed_lines)
    print('tokenizer_small.pickle')
    t = load_pickle_object('tokenizer_small.pickle')
    vocab_size = len(t.word_index) + 1
    print("Vocab size=", vocab_size)

    encoded_docs = t.texts_to_sequences(processed_lines)

    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='pre', truncating='pre')
    padded_docs = pad_sequences(padded_docs, maxlen=max_length+1, padding='post', truncating='pre')
    print(padded_docs.shape)
    return padded_docs, vocab_size

def get_padded_string(line, max_length=64): 
    
    processed_lines=[]
    n=len(processed_lines)
    if len(line)>0:
        newString = line.lower()
        newString = re.sub(r"'s\b","",newString)
        newString = re.sub("[^a-zA-Z 0-9]", "", newString)
        if len(newString.strip()) > 0:
            processed_lines.append(newString.strip())
    if VERBOSE:
        print("Number of sentences", "=",len(processed_lines)-n)

    t = load_pickle_object('tokenizer.pickle')
    vocab_size = len(t.word_index) + 1
    if VERBOSE:
        print("Vocab size=", vocab_size)

    encoded_docs = t.texts_to_sequences(processed_lines)

    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='pre', truncating='pre')
    padded_docs = pad_sequences(padded_docs, maxlen=max_length+1, padding='post', truncating='pre')
    if VERBOSE:
        print(padded_docs.shape)
    return padded_docs, vocab_size

def get_embedding_matrix():
    return load_pickle_object('embedding_matrix.pickle')

def get_tokenizer():
    return load_pickle_object('tokenizer.pickle')

def get_vocab_size():
    return len(load_pickle_object('tokenizer.pickle').word_index)+1

def decode_string(encoded_string):
    t = load_pickle_object('tokenizer.pickle')
    return t.sequences_to_texts(encoded_string)
