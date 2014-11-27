import re
import nltk.data
import sys
from sklearn import svm
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk import word_tokenize

def text(fiel):
    lst = []
    for line in fiel:
        lst.append(line)
    return lst

def numPunctuation(tokenList):
    punctuationList = [',' , '.' , ';', '!', '@', '#', '$', '%', '&', '*', '(', ')', ':' ,'?' , '/' , ':']

    numPunctuation = 0

    for token in tokenList:
        for char in token:
            if char in punctuationList:
                numPunctuation += 1

    return numPunctuation

def containsDate(tokenList):
    p = re.compile('\d?\d/\d?\d/\d?\d?\d\d')
    for token in tokenList:
        if p.search(token) != None:
            return True

    return False

def preProcess(message):
    """lowercase and tokenize input"""
    contains = False
    if '13/4/04' in message:
        print message
        contains = True
    message_tok = word_tokenize(message)
    message_tok = [w.lower() for w in message_tok]
    if contains:
        print 'found in preprocess'
        print message_tok
    return message_tok

def categorize(lst):
    hamlist, spamlist = [], []
    for x in lst:
        if x.startswith('spam'):
            spamlist.append(x[5:])
        elif x.startswith('ham'):
            hamlist.append(x[4:])
    return hamlist, spamlist

def main():

    textfile = open('SMSSpamCollection')
    lst = text(textfile)
    print(lst[0])
    hamlist, spamlist = categorize(lst)
    print(hamlist[0])
    print(spamlist[0])
    print(len(hamlist), len(spamlist))
    print(5574 - (len(hamlist)+len(spamlist)))

    hamProcessed = [preProcess(h) for h in hamlist]
    spamProcessed = [preProcess(s) for s in spamlist]
    
    spamWithDate = [s for s in spamProcessed if containsDate(s) ]
    hamWithDate = [h for h in hamProcessed if containsDate(h) ]
    
    print len(spamWithDate)
    print len(hamWithDate)

    print numPunctuation(preProcess(hamlist[0]))

main()