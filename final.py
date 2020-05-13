import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

################Stage-1: Sentence Level Classification

df_train = pd.read_csv('ngramsTrain.csv',header=None)
df_test = pd.read_csv('ngramsTest.csv',header=None)

#Encoding 9 classes for classification
mapping = {"bn_en_":0,"en_":1,"gu_en_":2,"hi_en_":3,"kn_en_":4,"ml_en_":5,"mr_en_":6,"ta_en_":7,"te_en_":8}
classes = ["bn_en_","en_","gu_en_","hi_en_","kn_en_","ml_en_","mr_en_","ta_en_","te_en_"]
languages = ["bengali","english","gujarati","hindi","kannada","malayalam","marathi","tamil","telugu"]


df_train = df_train.replace(mapping)
df_test = df_test.replace(mapping)

y_train = df_train[0]
x_train = df_train[1]
y_test = df_test[0]
x_test = df_test[1]

cv = CountVectorizer()
cv.fit(x_train)

new_x = cv.transform(x_train)
train_dataset = new_x.toarray()

######Naive Bayes Classifier
nb = MultinomialNB()
nb.fit(train_dataset,y_train)
######MaxEntropy i.e., Multi Class Logistic Regression
lg = LogisticRegression(random_state=0)
lg.fit(train_dataset,y_train)

new_x_test = cv.transform(x_test)
y_pred = nb.predict(new_x_test)
y1_pred = lg.predict(new_x_test)

print("F1 Score of Naive bayes for sentence classifier is ",metrics.accuracy_score(y_test,y_pred))      
print("F1 Score of Logistic Regression for sentence classifier is ",metrics.accuracy_score(y_test,y1_pred))

#########Testing with a new sentecnce

def ngram_generator(n,word):
    i=0
    n_grams=''
    j=1
    while(j<=n):
        i=0
        while(i<=len(word)-j):
            n_grams+=word[i:i+j]+' '
            i+=1
        j+=1
    return n_grams

##Performed for the commented test sentence.
# It can be any with restriction that only 2 mixed languages with english as fixed mixing language.

# test_sentence = "apna time aayega" 

test_sentence = input("Enter a sentence to predict its tags: ")
test_ngrams = ''
for item in test_sentence.split():
    test_ngrams +=ngram_generator(5,item)

lis = []
lis.append(test_ngrams)
lis = pd.DataFrame(lis)
final = cv.transform(lis[0])

y_final = nb.predict(final)        #Applying naive bayes to predict class label for input sentence.
y1_final = lg.predict(final)
label = y_final[0]

#print(y1_final)
print("For given sentence, predicted class label is ",classes[y_final[0]])

# Hence Naive Bayes Classifier and Logistic classifiers classified the given sentence
#with classlabel
#Now calling binary classifier 

################Stage 2 : Word level classification.

# Forming individual dataframes from Vocabularies.
en_df = pd.read_csv('eng2.txt',header=None)
lis = []
for i in range(len(en_df)):
    lis.append(1)
t = en_df[0]
en_df[0] = lis
en_df[1] = t

te_df = pd.read_csv('telugu.txt',header=None)
for i in range(len(te_df)):
    te_df[0][i] = ngram_generator(5,te_df[0][i])
lis = []
for i in range(len(te_df)):
    lis.append(8)
t = te_df[0]
te_df[0] = lis
te_df[1] = t

hi_df = pd.read_csv('hindiW.txt',header=None)
for i in range(len(hi_df)):
    hi_df[0][i] = ngram_generator(5,hi_df[0][i])
lis = []
for i in range(len(hi_df)):
    lis.append(3)
t = hi_df[0]
hi_df[0] = lis
hi_df[1] = t

ta_df = pd.read_csv('tamil.txt',header=None)
for i in range(len(ta_df)):
    ta_df[0][i] = ngram_generator(5,ta_df[0][i]) 
lis = []
for i in range(len(ta_df)):
    lis.append(7)
t = ta_df[0]
ta_df[0] = lis
ta_df[1] = t

#mr_df = pd.read_csv('maratiW.txt',header=None)
#for i in range(len(mr_df)):
#    mr_df[0][i] = ngram_generator(5,mr_df[0][i]) 

kn_df = pd.read_csv('kannadaW.txt',header=None)
for i in range(len(kn_df)):
    kn_df[0][i] = ngram_generator(5,kn_df[0][i])  
lis = []
for i in range(len(kn_df)):
    lis.append(4)
t = kn_df[0]
kn_df[0] = lis
kn_df[1] = t

gu_df = pd.read_csv('gujaratiW.txt',header=None)
for i in range(len(gu_df)):
    gu_df[0][i] = ngram_generator(5,gu_df[0][i])
lis = []
for i in range(len(gu_df)):
    lis.append(2)
t = gu_df[0]
gu_df[0] = lis
gu_df[1] = t

ml_df = pd.read_csv('malayalamW.txt',header=None)
for i in range(len(ml_df)):
    ml_df[0][i] = ngram_generator(5,ml_df[0][i])
lis = []
for i in range(len(ml_df)):
    lis.append(5)
t = ml_df[0]
ml_df[0] = lis
ml_df[1] = t

be_df = pd.read_csv('bengaliW.txt',header=None)
for i in range(len(be_df)):
    be_df[0][i] = ngram_generator(5,be_df[0][i])
lis = []
for i in range(len(be_df)):
    lis.append(0)
t = be_df[0]
be_df[0] = lis
be_df[1] = t
 
nb1 = MultinomialNB()
cv1 = CountVectorizer()

##Using same function for building 8 binary classifiers based on utput from upper Stage.
def binary_classifier(val):
    if val==8:
        lang_df = te_df
    elif val == 3:
        lang_df = hi_df
    elif val == 7:
        lang_df = ta_df
    elif val == 6:
        lang_df = mr_df 
    elif val == 4:
        lang_df = kn_df
    elif val == 2:
        lang_df = gu_df
    elif val == 5:
        lang_df = ml_df
    else:
        lang_df = be_df
    df = pd.concat([en_df,lang_df])
    X = df[1]
    y = df[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    cv1.fit(X_train)
    X_train = cv1.transform(X_train)
    nb1.fit(X_train,y_train)
    X_test = cv1.transform(X_test)
    y_pred = nb1.predict(X_test)
    print("F1 score of ",classes[label]," binary classifier is ",metrics.accuracy_score(y_test,y_pred))

test_list = test_sentence.split()
res = []
for item in test_list:
    n_grams = ngram_generator(5,item)
    lis = []
    lis.append(n_grams)
    lis = pd.DataFrame(lis)
    binary_classifier(label)
    lis = cv1.transform(lis[0])
    y_final = nb1.predict(lis)
    res.append(languages[y_final[0]])

print("Final word tags of given sentence are ", res)
#print(res)
#0:en, 1:te, 2:hi, 3:ta, 4:mr, 5:kn, 6:gu, 7:ml, 8:be
#Hence predicted tags for the given sentence are hi,en,hi

#apna - hi
#time - en
#aayega - hi
#Hence model is giving correct results.
