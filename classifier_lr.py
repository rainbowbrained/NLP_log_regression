from typing import List, Any
from random import random
import math
import numpy as np
from scipy.sparse import csr_matrix

# preprocessing: lowering and splitting
def preprocessing(text):
    text = text.lower()
    stripped_text = ''
    
    for c in text:
        if ((not c.isalnum()) and (c != ' ') and (len(c.encode(encoding='utf_8')) == 1)): 
            stripped_text = stripped_text + ' ' + c + ' '
        else:
            stripped_text += c
    return stripped_text
    
#tokenization: getting stuff together
def tokenization(text):
    q = text.split(' ')
    while ('' in q):
        q.remove('')
    return q

def append_dict(d, text):
    for i in range(len(text)):
        if (text[i] not in d):
            d.append(text[i])
    return


def generate_ngram (s, n): #s is a list of words. returns a list of collocations
    n -= 1
    tmp = []
    for j in range(n + 1):
            tmp.append(s[j])
    
    for i in range(len(s)):
        k = 0
        for j in range(n):
            if (i + j + 1 < len(s)):
                s[i] += ' '
                s[i] += s[i + j + 1]
            else:
                s[i] += ' '
                s[i] += tmp[k]
                k += 1
                
#weight is a vector
def sigmoid(weight, x): #If X is a vector - return number. X is a matrix - return vector
    b = x.dot(weight.T)
    b = 1 + np.exp(-b)
    return np.array(1/b)

def weight_init (x):
    a = np.zeros(x)
    return a

# Цикл по примерам. вычисляет оценочную функцию и точность
def direct_pass_cycle (X, y, w, hyperparam):
    loss = 0
    grad = np.arange(X.shape[1]) #same size as amount of features
    print("grad = ", grad)
    for i in range(X.shape[0]): #iterate over documents
        
        if (i % 100 == 0): #to control the training process
            print (i)
        
        h = sigmoid (w, X[i].toarray()) #same size as y = amount of documents
        grad = (h - y[i])*X[i]
        
        w = w - np.multiply(grad, hyperparam)
        if ((h != 1) and (h != 0)):
            loss += y[i]* math.log(h) + (1-y[i])* math.log(1-h)
            if (i == 0):
                print ("Loss rght after init: ", loss)
        
    loss = -1/X.shape[0]*loss
    precision = np.sum(abs(h - y))/X.shape[0]
    print ("Precision = ",precision)
    return loss

# Без цикла по примерам. вычисляет оценочную функцию и точность
def direct_pass (X, y, w, hyperparam):
    h = sigmoid(w, X)
    grad = new_grad(h, X, y)
    print ("grad = ", grad)
    w = update_weight(w, grad, 0)
    y = np.array(y)
    loss = -1/X.shape[0]*np.sum((y*np.log(h) + (1-y)*np.log(1-h)))
    precision = 1 - np.sum(abs(h - y))/X.shape[0]
    print ("Precision = ",precision)
    return loss

def update_weight (w, grad, hyperparam):
        regularization = 0.001
        return w - grad*hyperparam + regularization*w

def update_weight_momentum (w, grad, hyperparam, mom, gam):
    mom = gam*mom + hyperparam*grad
    return np.array(w - mom)
    
    
def new_grad (h, x, y):
    grad = x.transpose().dot(np.array(h-y))
    return grad/x.shape[0]
        
def mini_batch_gd (X, y, w, n_epoch, learning_rate):
    batch_size = 1000
    batch_cur = 0
    gamma = 0.1
    momentum = np.zeros(X.shape[1])
    flag = 1
    for i in range (n_epoch):
        h = sigmoid(w, X[batch_cur*batch_size:(batch_cur+1)*batch_size])
        grad = new_grad(h, X[batch_cur*batch_size:(batch_cur+1)*batch_size], y[batch_cur*batch_size:(batch_cur+1)*batch_size])
        w = update_weight(w, grad, learning_rate)
        precision = np.sum(abs(h - y[batch_cur*batch_size:(batch_cur+1)*batch_size]))/batch_size
        
        if ((i == 8000)|(i == 2000)|(i == 800)|(i == 5500)|(i == 10000)):
            learning_rate /= 2
        
        if (i % 50 == 0):
            h = sigmoid(w,X)
            precision = np.sum(abs(h - y))/X.shape[0]
            #print ("epoch № ", i)
            #print ("Precision in mini batch gd = ", 1 - precision)
            
            loss = -1/X.shape[0]*np.sum((y*np.log(0.001 + h) + (1-y)*np.log(1.001 - h)))
            
        if (batch_cur + 2 > X.shape[0]/batch_size):
            batch_cur = 0
        else: 
            batch_cur += 1
        if (precision < 0.01):
            break
    
    h = sigmoid(w,X)
    precision = 1 - np.sum(abs(h - y))/X.shape[0]
    print ("Precision in mini batch gd = ", precision)
    return w

def batch_gd (X, y, w, n_epoch, learning_rate):
    tmp = [0.0]
    batch_size = 2000
    batch_cur = 0
    flag = 1
    for i in range (n_epoch):
        h = sigmoid(w, X)
        grad = new_grad(h, X, y)
        w = update_weight(w, grad, learning_rate)
        precision = np.sum(abs(h - y))/X.shape[0]
        
        if (i % 100 == 0):
            tmp.append(precision)
            print ("epoch № ", i)
            print ("Precision in batch gd = ", 1 - precision)
            
            loss = -1/X.shape[0]*np.sum((y*np.log(0.001 + h) + (1-y)*np.log(1.001 - h)))
            tmp2.append(loss)
            
        if (i>500*flag):
            print ("mean/precision ", np.array(tmp)[-5:].mean()/precision)
            if (np.array(tmp)[-5:].mean()/precision < 1.1):
                learning_rate /= 2
                print ("Learning rate = ", learning_rate)
                flag += 3
                
        if (precision < 0.001):
            break

    precision = 1 - np.sum(abs(h - y))/X.shape[0]
    print ("Precision in batch gd = ", precision)
    return w


def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    
    doc_amount = len(train_texts) # amount of documents in train sample
    y_labels = []
    
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}

    for i in range(doc_amount):
        new_text = tokenization(preprocessing(train_texts[i]))
        generate_ngram(new_text, 2)
        new_text.insert(0, '')
        
        if (train_labels[i] == 'neg'):
            y_labels = y_labels + [0]
        else:
            y_labels = y_labels + [1]
            
        for term in new_text:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        #indices = list(set(indices))
        #data = data[:len(indices)]
        indptr.append(len(indices))
    matrix_all = csr_matrix((data, indices, indptr) )
    #matrix_all = matrix_all.log1p()
    y_labels = np.array(y_labels)

    print("------------------------------------------")
    
    n_epochs = 10000
    w = weight_init(len(vocabulary))
    w = mini_batch_gd(matrix_all, y_labels, w, n_epochs, 1)
    
    return {
        'dictionary' : vocabulary, 'weights' : w
    }


def classify(texts: List[str], params: Any) -> List[str]:
    
    vocabulary = params['dictionary']
    d1 = len(vocabulary)
    weights = params['weights']
    texts_labels = []
    indptr = [0]
    indices = []
    data = []
    for i in range(len(texts)):
        
        text_i = preprocessing(texts[i])
        text_i = tokenization(text_i)
        generate_ngram(text_i, 2)
        text_i.insert(0, '')
        
        if (i % 1000 == 0): 
            print (i)
        for term in text_i:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
        
    matrix_all = csr_matrix((data, indices, indptr) )
    d2 = len(vocabulary)
    
    tmp2 = np.zeros(d2).tolist()
    weights = weights.tolist()
    weights.extend(tmp2)
    weights = np.array(weights)
    
    h = sigmoid(weights[:matrix_all.shape[1]], matrix_all)
    
    for i in range(len(h)):
        if (h[i] > 0.5):
            texts_labels.append('pos')
        else:
            texts_labels.append('neg')
    return texts_labels
