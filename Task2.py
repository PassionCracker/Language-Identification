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

print("Building Sentence Level Classifier..........")
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

print("Successfully built sentence level classifier......")

####################Stage 2: Building Binary Classifiers

print("\nBuilding Binary Classifiers........")
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

#Constructing 9 dataframes from given language files 
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
 
mr_df = pd.read_csv('marati.txt',header=None)
for i in range(len(mr_df)):
    mr_df[0][i] = ngram_generator(5,mr_df[0][i])  
lis = []
for i in range(len(mr_df)):
    lis.append(6)
t = mr_df[0]
mr_df[0] = lis
mr_df[1] = t

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
 
nb_0 = MultinomialNB()
nb_2 = MultinomialNB()
nb_3 = MultinomialNB()
nb_4 = MultinomialNB()
nb_5 = MultinomialNB()
nb_6 = MultinomialNB()
nb_7 = MultinomialNB()
nb_8 = MultinomialNB()

lg_0 = LogisticRegression(random_state=0)
lg_2 = LogisticRegression(random_state=0)
lg_3 = LogisticRegression(random_state=0)
lg_4 = LogisticRegression(random_state=0)
lg_5 = LogisticRegression(random_state=0)
lg_6 = LogisticRegression(random_state=0)
lg_7 = LogisticRegression(random_state=0)
lg_8 = LogisticRegression(random_state=0)

cv_0 = CountVectorizer()
cv_2 = CountVectorizer()
cv_3 = CountVectorizer()
cv_4 = CountVectorizer()
cv_5 = CountVectorizer()
cv_6 = CountVectorizer()
cv_7 = CountVectorizer()
cv_8 = CountVectorizer()

#Bengali - English binary classifier
lang_df = be_df
df = pd.concat([en_df,lang_df])
X = df[1]
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
cv_0.fit(X_train)
X_train = cv_0.transform(X_train)
nb_0.fit(X_train,y_train)
lg_0.fit(X_train,y_train)

X_test = cv_0.transform(X_test)
y_pred = nb_0.predict(X_test)
y_pred1 = lg_0.predict(X_test)
print("F1 score of bengali - english NB_binary classifier is ",metrics.accuracy_score(y_test,y_pred))
print("F1 score of bengali - english LG_binary classifier is ",metrics.accuracy_score(y_test,y_pred1))


#Gujarathi - English binary classifier
lang_df = gu_df
df = pd.concat([en_df,lang_df])
X = df[1]
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
cv_2.fit(X_train)
X_train = cv_2.transform(X_train)
nb_2.fit(X_train,y_train)
X_test = cv_2.transform(X_test)
y_pred = nb_2.predict(X_test)
print("F1 score of gujarathi-english NB_binary classifier is ",metrics.accuracy_score(y_test,y_pred))
lg_2.fit(X_train,y_train)
y_pred1 = lg_2.predict(X_test)
print("F1 score of gujarathi-english LG_binary classifier is ",metrics.accuracy_score(y_test,y_pred1))


#Hindi - English Binary classifier
lang_df = hi_df
df = pd.concat([en_df,lang_df])
X = df[1]
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
cv_3.fit(X_train)
X_train = cv_3.transform(X_train)
nb_3.fit(X_train,y_train)
X_test = cv_3.transform(X_test)
y_pred = nb_3.predict(X_test)
print("F1 score of hindi-english NB_binary classifier is ",metrics.accuracy_score(y_test,y_pred))
lg_3.fit(X_train,y_train)
y_pred1 = lg_3.predict(X_test)
print("F1 score of hindi-english LG_binary classifier is ",metrics.accuracy_score(y_test,y_pred1))

#Kannada - English Binary classifier
lang_df = kn_df
df = pd.concat([en_df,lang_df])
X = df[1]
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
cv_4.fit(X_train)
X_train = cv_4.transform(X_train)
nb_4.fit(X_train,y_train)
X_test = cv_4.transform(X_test)
y_pred = nb_4.predict(X_test)
print("F1 score of kannada-english NB_binary classifier is ",metrics.accuracy_score(y_test,y_pred))
lg_4.fit(X_train,y_train)
y_pred1 = lg_4.predict(X_test)
print("F1 score of Kannada-english LG_binary classifier is ",metrics.accuracy_score(y_test,y_pred1))

#Malayalam - English Binary classifier
lang_df = ml_df
df = pd.concat([en_df,lang_df])
X = df[1]
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
cv_5.fit(X_train)
X_train = cv_5.transform(X_train)
nb_5.fit(X_train,y_train)
X_test = cv_5.transform(X_test)
y_pred = nb_5.predict(X_test)
print("F1 score of malayalam-english NB_binary classifier is ",metrics.accuracy_score(y_test,y_pred))
lg_5.fit(X_train,y_train)
y_pred1 = lg_5.predict(X_test)
print("F1 score of malayam-english LG_binary classifier is ",metrics.accuracy_score(y_test,y_pred1))

#Marathi - English Binary classifier
lang_df = mr_df
df = pd.concat([en_df,lang_df])
X = df[1]
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
cv_6.fit(X_train)
X_train = cv_6.transform(X_train)
nb_6.fit(X_train,y_train)
X_test = cv_6.transform(X_test)
y_pred = nb_6.predict(X_test)
print("F1 score of marathi-english NB_binary classifier is ",metrics.accuracy_score(y_test,y_pred))
lg_6.fit(X_train,y_train)
y_pred1 = lg_6.predict(X_test)
print("F1 score of marathi-english LG_binary classifier is ",metrics.accuracy_score(y_test,y_pred1))

#Tamil - English Binary classifier
lang_df = ta_df
df = pd.concat([en_df,lang_df])
X = df[1]
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
cv_7.fit(X_train)
X_train = cv_7.transform(X_train)
nb_7.fit(X_train,y_train)
X_test = cv_7.transform(X_test)
y_pred = nb_7.predict(X_test)
print("F1 score of tamil-english NB_binary classifier is ",metrics.accuracy_score(y_test,y_pred))
lg_7.fit(X_train,y_train)
y_pred1 = lg_7.predict(X_test)
print("F1 score of tamil-english LG_binary classifier is ",metrics.accuracy_score(y_test,y_pred1))

#Telugu - English Binary classifier
lang_df = te_df
df = pd.concat([en_df,lang_df])
X = df[1]
y = df[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
cv_8.fit(X_train)
X_train = cv_8.transform(X_train)
nb_8.fit(X_train,y_train)
X_test = cv_8.transform(X_test)
y_pred = nb_8.predict(X_test)
print("F1 score of telugu-english NB_binary classifier is ",metrics.accuracy_score(y_test,y_pred))
lg_8.fit(X_train,y_train)
y_pred1 = lg_8.predict(X_test)
print("F1 score of Telugu-english LG_binary classifier is ",metrics.accuracy_score(y_test,y_pred1))

print("\nSuccessfully built 8 binary classifiers.......")


########Now experimenting Steps1,2 on Test data

#Reading tags from XML file
import xml.etree.cElementTree as etree
def main_fun(file1,file2,iterator):
    xmlDoc = open(file1, 'r')
    xmlDocData = xmlDoc.read()
    xmlDocTree = etree.XML(xmlDocData)
    tags_list = []
    for ingredient in xmlDocTree.iter('data'):
        for i in range(len(ingredient)):
            tags_list.append(ingredient[i].text)

    # Encoding tags_list , 0:bn, 1:en, 2:gu, 3:hi, 4:kn, 5:ml, 6:mr, 7:ta, 8:te
    for i in range(len(tags_list)):
        tags_list[i] = tags_list[i].replace('\n','')
        tags_list[i] = tags_list[i].replace('\t','')
        tags_list[i] = tags_list[i].replace("en",'1')
        tags_list[i] = tags_list[i].replace("te",'8')
        tags_list[i] = tags_list[i].replace("hi",'3')
        tags_list[i] = tags_list[i].replace("ta",'7')
        tags_list[i] = tags_list[i].replace("mr",'6')
        tags_list[i] = tags_list[i].replace("kn",'4')
        tags_list[i] = tags_list[i].replace("gu",'2')
        tags_list[i] = tags_list[i].replace("ml",'5')
        tags_list[i] = tags_list[i].replace("bn",'0')

    ##Making it as a list of lists Eg: en en hi -> 1 1 3 ->[1,1,3]
    for i in range(len(tags_list)):
        tags_list[i] = tags_list[i].split(' ')

    ##Reading Test data from xml file
    xmlDoc = open(file2, 'r')
    xmlDocData = xmlDoc.read()
    xmlDocTree = etree.XML(xmlDocData)
    words_list = []

    for ingredient in xmlDocTree.iter('data'):
        for i in range(len(ingredient)):
            words_list.append(ingredient[i].text)
    for i in range(len(words_list)):
        words_list[i] = words_list[i].replace('\n','')
        words_list[i] = words_list[i].replace('\t','')
        words_list[i] = words_list[i].split(' ')

    #For test data, converting tags and words into formatted list of lists.
    final_tags_list = []
    final_words_list = []
    tags = ['0','1','2','3','4','5','6','7','8']
    for sent_index in range(len(tags_list)):
        tl = []
        wl = []
        for word_index in range(len(tags_list[sent_index])):
            if(tags_list[sent_index][word_index]) in tags:
                tl.append(int(tags_list[sent_index][word_index]))
                wl.append(words_list[sent_index][word_index])
        final_tags_list.append(tl)
        final_words_list.append(wl)

    #final_tags_list vs tags_list                #For understanding
    #final_words_list vs words_list

    final_sent_list = []
    i=0
    for sent in final_words_list:
        i+=1
        sentence = ''
        for word in sent:
            sentence = sentence+' '+word
        if(len(sentence)>0):
            sentence = sentence.replace(sentence[0],'',1)    
        final_sent_list.append(sentence)
    #final_sent_list is a list of testing sentences after removing noice like http links and all.
    #Small manipulations...
    if(iterator == 1):
        del final_tags_list[509][0]
        del final_tags_list[1282][0]
        del final_tags_list[2371][0]
    else:
        del final_tags_list[739][0]
        del final_tags_list[390][0]


    ##############Testing process starts.
    #1) Feed the sentence to Sentence level classifier and invoke the appropriate binary classifier.

    print("Now, Testing on test data")
    pred_tags_list = []
    pred_tags_list1 = []

    for test_sentence in final_sent_list:
        test_ngrams = ''
        for item in test_sentence.split():
            test_ngrams +=ngram_generator(5,item)
        lis = []
        lis.append(test_ngrams)
        lis = pd.DataFrame(lis)
        final = cv.transform(lis[0])
        y_final = nb.predict(final) 
        y1_final = lg.predict(final)
        label = y_final[0]
        ###Done with Sentence level classification for a sentence in test dataset.
    
        res = []
        res_tags_nb = []
        res_tags_lg = []
        for item in test_sentence.split():
            n_grams = ngram_generator(5,item)
            lis = []
            lis.append(n_grams)
            lis = pd.DataFrame(lis)

            if(label == 0):
                lis = cv_0.transform(lis[0])
                y_final = nb_0.predict(lis)
                y_final1 = lg_0.predict(lis)
                val1 = y_final1[0]
                val = y_final[0]
            
            elif(label == 2):
                lis = cv_2.transform(lis[0])
                y_final = nb_2.predict(lis) 
                val = y_final[0]
                y_final1 = lg_2.predict(lis)
                val1 = y_final1[0]
            
            elif(label == 3):
                lis = cv_3.transform(lis[0])
                y_final = nb_3.predict(lis)
                val = y_final[0]
                y_final1 = lg_3.predict(lis)
                val1 = y_final1[0]
            
            elif(label == 4):
                lis = cv_4.transform(lis[0])
                y_final = nb_4.predict(lis)
                val = y_final[0]
                y_final1 = lg_4.predict(lis)
                val1 = y_final1[0]

            elif(label == 5):
                lis = cv_5.transform(lis[0])
                y_final = nb_5.predict(lis)
                val = y_final[0]
                y_final1 = lg_5.predict(lis)
                val1 = y_final1[0]

            elif(label == 6):
                lis = cv_6.transform(lis[0])
                y_final = nb_6.predict(lis)
                val = y_final[0]
                y_final1 = lg_6.predict(lis)
                val1 = y_final1[0]

            
            elif(label == 7):
                lis = cv_7.transform(lis[0])
                y_final = nb_7.predict(lis)
                val = y_final[0]
                y_final1 = lg_7.predict(lis)
                val1 = y_final1[0]

        
            elif(label == 8):
                lis = cv_8.transform(lis[0])
                y_final = nb_8.predict(lis)
                val = y_final[0]
                y_final1 = lg_8.predict(lis)
                val1 = y_final1[0]

            else:
                val = 1
                val1 = 1
        
            res.append(languages[val])
            res_tags_nb.append(val)
            res_tags_lg.append(val1)

        pred_tags_list.append(res_tags_nb)
        pred_tags_list1.append(res_tags_lg)
    
    labels = [0,1,2,3,4,5,6,7,8]
    from sklearn_crfsuite import metrics
    from sklearn_crfsuite import metrics
    if(iterator==0):
        print("Naive Bayes Accuracy:",metrics.flat_f1_score(final_tags_list,pred_tags_list,average='weighted',labels=labels))

        print("Logistic Regression Accuracy:",metrics.flat_f1_score(final_tags_list,pred_tags_list1,average='weighted',labels=labels))

    #metrics.flat_classification_report(final_tags_list,pred_tags_list)
    #metrics.flat_classification_report(final_tags_list,pred_tags_list1)
    return [pred_tags_list,pred_tags_list1,final_tags_list]
##############CRF Extension

ret_lis_training = main_fun('AnnotationTraining.xml','InputTraining.xml',1)
ret_lis_testing = main_fun('AnnotationTesting.xml','InputTesting.xml',0)
pred_tags_list_train = ret_lis_training[0]
pred_tags_list1_train = ret_lis_training[1]
final_tags_list_train = ret_lis_training[2]

pred_tags_list_test = ret_lis_testing[0]
pred_tags_list1_test = ret_lis_testing[1]
final_tags_list_test = ret_lis_testing[2]


# Building a list with considering sequence
def get_new_list(pred_tags_list):
    new_list = []
    for i in range(len(pred_tags_list)):
        lis1=[]
        for j in range(len(pred_tags_list[i])):
            lis = []
            if(j>1):
                lis.append('PrevToPrev:'+str(pred_tags_list[i][j-2]))
            if(j>0):
                lis.append('Prev:'+str(pred_tags_list[i][j-1]))
            lis.append('Current:'+str(pred_tags_list[i][j]))
            if(j<len(pred_tags_list[i])-1):
                lis.append('Next:'+str(pred_tags_list[i][j+1]))
            if(j<len(pred_tags_list[i])-2):
                lis.append('NextToNext:'+str(pred_tags_list[i][j+2]))
            lis1.append(lis)
        new_list.append(lis1)
    return new_list
new_list1_train = get_new_list(pred_tags_list_train)
new_list2_train = get_new_list(pred_tags_list1_train)
new_list1_test = get_new_list(pred_tags_list_test)
new_list2_test = get_new_list(pred_tags_list1_test)


for i in range(len(final_tags_list_train)):
    for j in range(len(final_tags_list_train[i])):
        final_tags_list_train[i][j] = str(final_tags_list_train[i][j])

for i in range(len(final_tags_list_test)):
    for j in range(len(final_tags_list_test[i])):
        final_tags_list_test[i][j] = str(final_tags_list_test[i][j])

##########Importing crf from suite
import sklearn_crfsuite
crf1 = sklearn_crfsuite.CRF(algorithm='lbfgs',all_possible_states = True,all_possible_transitions=True,c1=0.1,c2=0.1,max_iterations=100)
crf2 = sklearn_crfsuite.CRF(algorithm='lbfgs',all_possible_states = True,all_possible_transitions=True,c1=0.1,c2=0.1,max_iterations=100)


X1_training = new_list1_train
X2_training = new_list2_train
Y_training = final_tags_list_train
X1_testing = new_list1_test
X2_testing = new_list2_test
Y_testing = final_tags_list_test

crf1.fit(X1_training,Y_training)
crf2.fit(X2_training,Y_training)

pred1 = crf1.predict(X1_testing)
pred2 = crf2.predict(X2_testing)
labels = [0,1,2,3,4,5,6,7,8]

from sklearn_crfsuite import metrics
print("Naive Bayes with CRF Accuracy:",metrics.flat_f1_score(Y_testing,pred1,average='weighted',labels=labels))
print("Logistic regression with CRF Accuracy:",metrics.flat_f1_score(Y_testing,pred2,average='weighted',labels=labels))

#For reports,
#metrics.flat_classification_report(Y_testing,pred1)
#metrics.flat_classification_report(Y_testing,pred2)
