import re
import nltk.data
import sys
from sklearn import svm
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk import word_tokenize
from sklearn.metrics import confusion_matrix

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

"""
    Given a list of list of tokens
    e.g. [['hi', 'there'], ['hi','bye']]
    it will return the vocabulary of the list

    ['hi,'there','bye']
"""
def generateVocabulary(tokenListList):
    vocab = []
    letterList = []
    allTokens = []

    pattern = re.compile(r'[a-z]+')

    for tokenList in tokenListList:
        for token in tokenList:
            """retain only [a-z] in token"""
            allMatches = pattern.findall(token)
            if len(allMatches) > 0:
                vocabWord = ''.join(allMatches)
                vocab.append(vocabWord)

    return set(vocab)



def containsDate(tokenList):
    p = re.compile('\d?\d/\d?\d/\d?\d?\d\d')
    for token in tokenList:
        if p.search(token) != None:
            return True

    return False

def preProcess(message):
    """lowercase and tokenize input"""

    message_tok = word_tokenize(message)
    message_tok = [w.lower() for w in message_tok]

    return message_tok

def categorize(lst):
    hamlist, spamlist = [], []
    for x in lst:
        if x.startswith('spam'):
            spamlist.append(x[5:])
        elif x.startswith('ham'):
            hamlist.append(x[4:])
    return hamlist, spamlist

def numChars(tokenList):
    totalLength = 0
    for token in tokenList:
        totalLength+=1
    return totalLength

def getFeatures(vocab, tokenList):
    features = {}

    tokenUnigrams = generateVocabulary([tokenList])
    for word in vocab:
        if word in tokenUnigrams:
            features['Unigram-' + word] = 1
        else:
            features['Unigram-' + word] = 0

    features['ContainsDate'] = containsDate(tokenList)
    features['numPunctuation'] = numPunctuation(tokenList)
    features['numChars'] = numChars(tokenList)

    return features

def main():

    if len(sys.argv) < 3:
        print "train file as first argument, test file as second"
        sys.exit()
    trainFileName = sys.argv[1]
    testFileName = sys.argv[2]

    """Process TRAIN file"""
    print "Processing train file"
    trainfile = open(trainFileName)
    lst_train = text(trainfile)
    hamlist_train, spamlist_train = categorize(lst_train)

    hamProcessedTrain = [preProcess(h) for h in hamlist_train]
    spamProcessedTrain = [preProcess(s) for s in spamlist_train]
    
    vocab = generateVocabulary(hamProcessedTrain + spamProcessedTrain)

    hamFeaturedTrain = [ (getFeatures(vocab, h), 0) for h in hamProcessedTrain]
    spamFeaturedTrain = [ (getFeatures(vocab, s), 1) for s in spamProcessedTrain]
    
    allTrainData = hamFeaturedTrain + spamFeaturedTrain
    svmDataTrain = [d[0] for d in allTrainData]
    svmAnsTrain = [d[1] for d in allTrainData]

    """Process TEST file """
    print "Processing test file"
    testfile = open(testFileName)
    lst_test = text(testfile)
    hamlist_test, spamlist_test = categorize(lst_test)

    hamProcessedTest = [preProcess(h) for h in hamlist_test]
    spamProcessedTest = [preProcess(s) for s in spamlist_test]

    hamFeaturedTest = [ (getFeatures(vocab, h), 0) for h in hamProcessedTest]
    spamFeaturedTest = [ (getFeatures(vocab, s), 1) for s in spamProcessedTest]
    
    allTestData = hamFeaturedTest + spamFeaturedTest
    svmDataTest = [d[0] for d in allTestData]
    svmAnsTest = [d[1] for d in allTestData]

    vec = DictVectorizer()
    svmTrainDataVectored = vec.fit_transform(svmDataTrain).toarray()
    svmTestDataVectored = vec.fit_transform(svmDataTest).toarray()

    svmClass = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 600)
    print "fitting SVM"
    test_predict = svmClass.fit(svmTrainDataVectored, svmAnsTrain).predict(svmTestDataVectored)

    cm = confusion_matrix(svmAnsTest, test_predict)

    print cm

main()