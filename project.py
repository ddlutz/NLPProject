import re
import nltk.data
import sys
from sklearn import svm
from sklearn import datasets
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk import word_tokenize
import sklearn.metrics

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

    vocabCount = {}

    for tokenList in tokenListList:
        for token in tokenList:
            """retain only [a-z] in token"""
            allMatches = pattern.findall(token)
            if len(allMatches) > 0:
                vocabWord = ''.join(allMatches)
                vocab.append(vocabWord)
                if vocabWord in vocabCount:
                    vocabCount[vocabWord] = vocabCount[vocabWord] +1
                else:
                    vocabCount[vocabWord] = 1

    return [w for w in vocabCount.keys()]


def containsDate(tokenList):
    p = re.compile('\d?\d/\d?\d/\d?\d?\d\d')
    for token in tokenList:
        if p.search(token) != None:
            return True

    return False
    
"""
def phoneNumber(msg):
    result = re.search(r"\d{5,12}", msg)
    if result == None:
        return False
    return True
"""
    
def phoneNumber(tokenList):
    p = re.compile(r"\d{5,12}")
    for token in tokenList:
        if p.search(token) != None:
            return True
    return False

"""
def emailAddress(msg):
    result = re.search(r"\w+@\w+.\w+", msg)
    if result == None:
        return False
    return True
"""

def emailAddress(tokenList):
    p = re.compile(r"\w+@\w+.\w+")
    #have to join b/c periods split up tokens
    allText = "".join(tokenList)
    if p.search(allText):
        return True
    else:
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
    features['emailAddress'] = emailAddress(tokenList)
    features['phoneNumber'] = phoneNumber(tokenList)

    return features

def getFeaturesFiltered(vocab, tokenList, filters):
    
    features = {}

    tokenUnigrams = generateVocabulary([tokenList])
    for word in vocab:
        if word in tokenUnigrams and (('Unigram-' + word) in filters):
            features['Unigram-' + word] = 1
        elif ('Unigram-' + word) in filters:
            features['Unigram-' + word] = 0

    if 'ContainsDate' in filters:
        features['ContainsDate'] = containsDate(tokenList)
    if 'numPuncutation' in filters:
        features['numPunctuation'] = numPunctuation(tokenList)
    if 'numChars' in filters:
        features['numChars'] = numChars(tokenList)
    if 'emailAddress' in filters:
        features['emailAddress'] = emailAddress(tokenList)
    if 'phoneNumber' in filters:
        features['phoneNumber'] = phoneNumber(tokenList)


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
    

    """Process TEST file """
    print "Processing test file"
    testfile = open(testFileName)
    lst_test = text(testfile)
    hamlist_test, spamlist_test = categorize(lst_test)

    hamProcessedTest = [preProcess(h) for h in hamlist_test]
    spamProcessedTest = [preProcess(s) for s in spamlist_test]

    hamFeaturedTest = [ (getFeatures(vocab, h), 0) for h in hamProcessedTest]
    spamFeaturedTest = [ (getFeatures(vocab, s), 1) for s in spamProcessedTest]

    numCorrect = 0
    feature = None

    allTrainData = hamFeaturedTrain + spamFeaturedTrain
    svmDataTrain = [d[0] for d in allTrainData]
    svmAnsTrain = [d[1] for d in allTrainData]

    allTestData = hamFeaturedTest + spamFeaturedTest
    svmDataTest = [d[0] for d in allTestData]
    svmAnsTest = [d[1] for d in allTestData]

    vec = DictVectorizer()
    svmTrainDataVectored = vec.fit_transform(svmDataTrain).toarray()
    svmTestDataVectored = vec.fit_transform(svmDataTest).toarray()
    
    print 'total # of features: '
    print len(svmTestDataVectored[0])
    svmClass = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 600)
    
    """
    Potential classifiers - unused, obvioulsy
    svm2 = svm.SVC(kernel = 'rbf', gamma=0.005, C = 600)
    svm3 = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 10)
    svm4 = svm.SVC(kernel = 'rbf', gamma=0.005, C = 10)
    svm5 = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 1)
    svm6 = svm.SVC(kernel = 'rbf', gamma=0.005, C = 1)
    svm7 = svm.SVC(kernel = 'rbf', gamma=0.00005, C = 600)
    svm8 = svm.SVC(kernel = 'rbf', gamma=0.00005, C = 10)
    svm9 = svm.SVC(kernel = 'rbf', gamma=0.00005, C = 1)
    
    After narrowing down gamma and C values, these svms were tested but showed
    the same results as the one used.
    
    svm10 = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 500)
    svm11 = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 700)
    """
    print "fitting SVM"
    test_predict = svmClass.fit(svmTrainDataVectored, svmAnsTrain).predict(svmTestDataVectored)

    """
    Uncomment below code to display 
    answers which were predicted incorrectly


    print "Num items in test ham: " + str(len(hamProcessedTest))
    print "Num itmes in test spam: " + str(len(spamProcessedTest))

    print "Num items in test ans: " + str(len(test_predict))
    for i in range(len(test_predict)):
        if test_predict[i] != svmAnsTest[i]:
            print "*******************"
            print "Test index wrong: " + str(i)
            if i >= len(hamFeaturedTest):
                spamIndex = i - len(hamProcessedTest)
                print "Index in spam: " + str(spamIndex)
                print spamProcessedTest[spamIndex]
            else:
                print hamFeaturedTest[i]
            print "*******************"
    """

    print sklearn.metrics.classification_report(svmAnsTest, test_predict)


    cm = sklearn.metrics.confusion_matrix(svmAnsTest, test_predict)

    print svmClass.get_params() #Print out params
    print cm                    #Print confusion matrix

    filters = []
    from sklearn.feature_selection import chi2, SelectKBest
    selector = SelectKBest(chi2).fit(svmTrainDataVectored, svmAnsTrain) # There is more to SelectKBest; cf. documentation
    top10 = selector.scores_.argsort()[::-1][:10]
    for f in top10:
        print '%.3e\t%s' % (selector.pvalues_[f], vec.get_feature_names()[f])
        filters.append(vec.get_feature_names()[f])

    hamFeaturedTrain = [ (getFeaturesFiltered(vocab, h,filters), 0) for h in hamProcessedTrain]
    spamFeaturedTrain = [ (getFeaturesFiltered(vocab, s,filters), 1) for s in spamProcessedTrain]

    hamFeaturedTest = [ (getFeaturesFiltered(vocab, ha, filters), 0) for ha in hamProcessedTest]
    spamFeaturedTest = [ (getFeaturesFiltered(vocab, sa, filters), 1) for sa in spamProcessedTest]


    allTrainData = hamFeaturedTrain + spamFeaturedTrain
    svmDataTrain = [d[0] for d in allTrainData]
    svmAnsTrain = [d[1] for d in allTrainData]

    allTestData = hamFeaturedTest + spamFeaturedTest
    svmDataTest = [d[0] for d in allTestData]
    svmAnsTest = [d[1] for d in allTestData]

    vec = DictVectorizer()
    svmTrainDataVectored = vec.fit_transform(svmDataTrain).toarray()
    svmTestDataVectored = vec.fit_transform(svmDataTest).toarray()


    svmClass = svm.SVC(kernel = 'rbf', gamma=0.0005, C = 600)
    print "fitting SVM again" 
    test_predict = svmClass.fit(svmTrainDataVectored, svmAnsTrain).predict(svmTestDataVectored)

    print sklearn.metrics.classification_report(svmAnsTest, test_predict)
    cm = sklearn.metrics.confusion_matrix(svmAnsTest, test_predict)

    print cm

main()
