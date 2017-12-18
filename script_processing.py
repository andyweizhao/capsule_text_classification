# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import re

def load_data(fname):
    data=[]
    label=[]
    with open(fname) as f:
        for line in f:
            line=line.lower()
            line=re.sub(r'[^\x00-\x7F]+','', line)
            line=line.replace('[','')
            line=line.replace(']','')
            line=line.replace('\'','')
            now=line.strip().split()
            label.append(int(now[0]))
            data.append(now[1:])
    return data,label

def preprocess_data(data):
    punctuations=['!','"','#','.','+',',','-','/',';',':','()','=','@','?',r'\\','{','}','~','|','[',']']
    stop_words = []
#    stop_words=[u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the',u'The']
    filter=punctuations+stop_words
    clean_data=[]
    for i in data:
        item=[]
        for j in i:
            if j not in filter:
                item.append(j)
        clean_data.append(item)
    return clean_data

def build_vocab(data, count_thr):
    counts = {}
    for _ in ('train','val','test'):
        for sent in data[_]:
            for w in sent:     
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:10])))

    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]
    vocab = [w for w,n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    sent_lengths = {}
    for _ in ('train','val','test'):
        for sent in data[_]:
            nw = len(sent)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len+1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

    if bad_count > 0:
    # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')
      
    for _ in ('train','val','test'):
        for i, sent in enumerate(data[_]):
            data[_][i] = [w if counts.get(w,0) > count_thr else 'UNK' for w in sent]
    return vocab

def encode_sentence(data, label, max_length):
    n = len(data)
#    Li = np.zeros((n, max_length + 2), dtype='uint32')
    Li = np.zeros((n, max_length), dtype='uint32')
    for j,s in enumerate(data):
        for k,w in enumerate(s):        
            if k < max_length:
#                Li[j,k+1] = wtoi[w]
                Li[j,k] = wtoi[w]
                
    return np.concatenate((np.expand_dims(label, axis=1), Li),axis=1)      

def load_word_vector(fname):
    model = {}
    with open(fname) as fin:
        for line_no, line in enumerate(fin):
            try:
                parts = line.strip().split(' ')
                word, weights = parts[0], parts[1:]
                model[word] = np.array(weights,dtype=np.float32)
            except:
                pass
    return model

if __name__ == "__main__":
    max_length = 30
    dataset = 'MR'
    
    x_train_orign, y_train = load_data(r'data/'+dataset+'/train.txt')
    x_dev_orign, y_dev = load_data(r'data/'+dataset+'/dev.txt')
    x_test_orign, y_test = load_data(r'data/'+dataset+'/test.txt')
    
    data = {}
    data['train'] = preprocess_data(x_train_orign)
    data['val'] = preprocess_data(x_dev_orign)
    data['test'] = preprocess_data(x_test_orign)
    
    count_thr = 0
    
    vocab = build_vocab(data, count_thr)
    
    itow = {i:w for i,w in enumerate(vocab)}#i+1
    wtoi = {w:i for i,w in enumerate(vocab)} 
    
    train_data = encode_sentence(data['train'], y_train, max_length)
    val_data = encode_sentence(data['val'], y_dev, max_length)
    test_data = encode_sentence(data['test'], y_test, max_length)
    
    model = load_word_vector('data/glove.6B.50d.txt')
    model_vocab = set(model.keys())
    
    paras = [model[itow[i]] if itow[i] in model_vocab else model['the'] for _, i in enumerate(itow)]
    paras = np.array(paras)