import os, re
import numpy as np
filenames = os.listdir(os.getcwd() + '/txt/')
len(filenames)

def sanitize(s):
    #Sanitize a string
    return re.sub(r'[a-zA-Z0-9]*[0-9]+[a-zA-Z0-9]*|[_-]+|[^\w]+|pdf|txt',' ',s)

sanitized_filenames = [sanitize(filename).strip() for filename in filenames]
training_keys = [x for x in sanitized_filenames if x != '']
training_filenames = np.array(filenames)[[(x != '') for x in sanitized_filenames]]
match_filenames = np.array(filenames)[[(x == '') for x in sanitized_filenames]]


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
vectorizer=TfidfVectorizer()
X = vectorizer.fit_transform(training_keys)
tf_transformer = TfidfTransformer(use_idf=False).fit(X)
names=vectorizer.get_feature_names()


train_files = {}
for filename in training_filenames:
    f = open('txt/' + filename,'r',encoding = "ISO-8859-1")
    train_files[filename]=f.read()
    f.close()

match_files = {}
for filename in match_filenames:
    f = open('txt/' + filename,'r',encoding = "ISO-8859-1")
    match_files[filename]=f.read()
    f.close()
    
    
name_vectorizer=CountVectorizer()
names = name_vectorizer.fit_transform(training_keys)
names=name_vectorizer.get_feature_names()    
    
vectorizer=TfidfVectorizer(vocabulary=names)
X = list(train_files.values())
y = list(train_files.keys())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
X_train_counts=vectorizer.fit_transform(X_train)
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)


from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(X_train_tf,y_train)

X_match=match_files.values()
X_match_counts=vectorizer.transform(X_match)
X_match_tfidf = tf_transformer.transform(X_match_counts)

predicted=clf.predict(X_match_tfidf)
for a,b in zip(match_files.keys(),predicted):
    print('%r => %s'% (a,re.sub('.pdf.txt','--',a) + b))
