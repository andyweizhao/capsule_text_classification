import numpy as np
import h5py
import re
import operator

import sys
reload(sys)
sys.setdefaultencoding('utf8')

from nltk import word_tokenize
from nltk.corpus import reuters
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")

def load_word_vector(fname, vocab):
    model = {}
    with open(fname) as fin:
        for line_no, line in enumerate(fin):
            try:
                parts = line.strip().split(' ')
                word, weights = parts[0], parts[1:]
                if word in vocab:                     
                    model[word] = np.array(weights,dtype=np.float32)
            except:
                pass
    return model

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab: 
               word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')  
            else:
               f.read(binary_len)
    return word_vecs

def line_to_words(line):
    words = map(lambda word: word.lower(), word_tokenize(line))
    tokens = words
    p = re.compile('[a-zA-Z]+')
    return list(filter(lambda token: p.match(token) and len(token) >= 3, tokens))        

def get_vocab(dataset):
    max_sent_len = 0
    word_to_idx = {}
    idx = 1
    for line in dataset:    
        words = line_to_words(line)
        max_sent_len = max(max_sent_len, len(words))
        for word in words:
            if not word in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    return max_sent_len, word_to_idx


#dataset = 'reuters_multilabel_all'
dataset = 'reuters_multilabel_new'

def load_data(padding=0, sent_len=300, w2i=None):
    
    """
        threshold = 0  all labels in test data
        threshold = 1  only multilabels in test data
    """
    threshold = 1 
    
    train_docs, train_cats, test_docs, test_cats = [], [], [], []
    
    popular_topics = set(['earn','acq','money-fx','grain','crude','trade','interest','ship','wheat','corn'])
    
    for doc_id in reuters.fileids():
        if doc_id.startswith("train"):
            if set(reuters.categories(doc_id)).issubset(popular_topics):
                train_docs.append(reuters.raw(doc_id))
                train_cats.append([cat for cat in reuters.categories(doc_id)])
#            train_cats.append(
#                [cats.index(cat) for cat in reuters.categories(doc_id)])
        else:
            if set(reuters.categories(doc_id)).issubset(popular_topics):
                test_docs.append(reuters.raw(doc_id))
                test_cats.append([cat for cat in reuters.categories(doc_id)])
    
    dataset = train_docs + test_docs
    max_sent_len, word_to_idx = get_vocab(dataset)
    if sent_len > 0:
        max_sent_len = sent_len      
    if w2i is not None:
        word_to_idx = w2i    

    train, train_label, test, test_label = [], [], [], []
        
    for i, line in enumerate(train_docs):
        words = line_to_words(line)
        y = train_cats[i]
        if len(y) > 1: # The examples which contain at least 1 label would be assigned to test data.
            test_docs.append(line)
            test_cats.append(y)
            continue           
        y = y[0]
        sent = [word_to_idx[word] for word in words if word in word_to_idx]
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        else:    
            sent.extend([0] * (max_sent_len + padding - len(sent)))
        train.append(sent)
        train_label.append(y)
    
    single_label = ['-1'] + list(set(train_label))
    num_classes = len(single_label)
    for i, l in enumerate(train_label):
        train_label[i] = single_label.index(l)
        
    for i, line in enumerate(test_docs):
        words = line_to_words(line)
        y = test_cats[i]    
        sent = [word_to_idx[word] for word in words if word in word_to_idx]
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        else:    
            sent.extend([0] * (max_sent_len + padding - len(sent)))
        if len(y) > threshold and set(y).issubset(single_label):
            test.append(sent)
            one_hot_y = np.zeros([num_classes],dtype=np.int32)
            for yi in y:
                one_hot_y[single_label.index(yi)]=1 
            test_label.append(one_hot_y)
        
    return single_label, word_to_idx, np.array(train), np.array(train_label), np.array(test), np.array(test_label)


single_label, word_to_idx, train, train_label, test, test_label = load_data(padding=0, sent_len=300, w2i=None)

with open(dataset + '_word_mapping.txt', 'w+') as embeddings_f:
   embeddings_f.write("*PADDING* 0\n")
   for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
      embeddings_f.write("%s %d\n" % (word, idx))

w2v_path = 'GoogleNews-vectors-negative300.bin'
#glove_path = 'glove.6B/glove.6B.300d.txt'

w2v = load_bin_vec(w2v_path, word_to_idx)
V = len(word_to_idx) + 1
print ('Vocab size:', V) 

def compute_embed(V,w2v):
    np.random.seed(1)
    embed = np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0])))
    for word, vec in w2v.items():
        embed[word_to_idx[word]] = vec # padding word is positioned at index 0
    return embed

embed_w2v = compute_embed(V, w2v)
#embed_glove = compute_embed(V, glove)

print ('train size:', train.shape)

filename = dataset + '.hdf5'
with h5py.File(filename, "w") as f:
    f["w2v"] = np.array(embed_w2v)
    f['train'] = train
    f['train_label'] = train_label
    f['test'] = test
    f['test_label'] = test_label

