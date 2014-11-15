def text(fiel):
    lst = []
    for line in fiel:
        lst.append(line)
    return lst

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

main()
