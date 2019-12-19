import pandas as pd
import numpy as np
from sklearn import preprocessing # for encoding labels


df = pd.read_csv('/Users/macbookair/documents/Semester 7/Natural Language Processing/Exam/folketinget_1993_2019_tokenized.csv')

'''
-------------------- Preprocessing -------------------
'''

# Keep only speeches from 20XX and later
df['Dato'] = df['Dato'].str.slice(start = 6, stop = 10)
df['Dato'] = df['Dato'].astype(int)
df = df[df['Dato'] > 2009]

# Unique speakers
speakers = df['Title']
unique_speakers = df['Title'].unique()

# Filter out formanden
df = df[~df.Title.isin(['formand', 'Henrik Dam Kristensen', 'Leif Mikkelsen', 'Næstformand', 'Pia Kjærsgaard', 'næstformand'])]

# Replace nan etc.
df['text'] = df.text.str.replace('nan' , '')
df['text'] = df.text.str.replace('tikke' , 'tak')
df['text'] = df.text.str.replace('matiasse' , 'mathias')


# Remove all speeches shorter than 80 characters
df=df[(df.text.astype(str).str.len()>80)]

# How many speeches each speaker has, how many parties that are etc.
counts = df['Title'].value_counts()
df['Parti'].unique()
df['Parti'].value_counts()

# Standardize party names and include the 9 biggest (Nye Borgerlige not annotated)
df.loc[df['Parti'] == 'Fælles', 'Name']
df.loc[df['Name'] == 'Pernille Vermund', 'Parti']

df['Parti'] = df['Parti'].replace({'Dansk': 'Dansk_Folkeparti'})
df['Parti'] = df['Parti'].replace({'DF': 'Dansk_Folkeparti'})
df['Parti'] = df['Parti'].replace({'Radikale': 'Radikale_Venstre'})
df['Parti'] = df['Parti'].replace({'Socialistisk': 'Socialistisk_Folkeparti'})
df['Parti'] = df['Parti'].replace({'Liberal': 'Liberal_Alliance'})
df['Parti'] = df['Parti'].replace({'V': 'Venstre'})
df['Parti'] = df['Parti'].replace({'Socialdemokraterne': 'Socialdemokratiet'})
df['Parti'] = df['Parti'].replace({'S': 'Socialdemokratiet'})
df['Parti'] = df['Parti'].replace({'Det': 'Det_Konservative_Folkeparti'})

df = df.loc[df['Parti'].isin(['Dansk_Folkeparti', 'Radikale_Venstre', 'Venstre', 'Socialdemokratiet', 'Socialistisk_Folkeparti', 'Liberal_Alliance', 'Det_Konservative_Folkeparti', 'Enhedslisten', 'Alternativet'])]

# Remove NaN from text and Parti and check that it worked
df = df.dropna(subset=['text', 'Parti'])
df['text'].isna().values.sum()

'''
# All sentences, remove nan
sents = []
for sent in df['text']:
    sents.append(sent)
clean_sents = [x for x in sents if str(x) != 'nan']

# Save all words in the list 'words'
words = []
for i in range(len(clean_sents)):
    sent = clean_sents[i]
    if sent != float :
        w = sent.split(' ')
        for word in w:
            words.extend(w)

# 1187401183 words (1 milliard 187 millioner 401183 ord)
print(len(words))

# Check most common words
from collections import Counter
most_common_words= [word for word, word_count in Counter(words).most_common(25)]
print(most_common_words)
'''

'''
-------------------- Binary Classification -------------------
'''

# Binary Classification liberal/socialist
df['Parti'] = df['Parti'].replace({'Det_Konservative_Folkeparti': 0})
df['Parti'] = df['Parti'].replace({'Dansk_Folkeparti': 0})
df['Parti'] = df['Parti'].replace({'Radikale_Venstre': 0})
df['Parti'] = df['Parti'].replace({'Venstre': 0})
df['Parti'] = df['Parti'].replace({'Socialdemokratiet': 1})
df['Parti'] = df['Parti'].replace({'Socialistisk_Folkeparti': 1})
df['Parti'] = df['Parti'].replace({'Liberal_Alliance': 0})
df['Parti'] = df['Parti'].replace({'Enhedslisten': 1})
df['Parti'] = df['Parti'].replace({'Alternativet': 1})

from sklearn.model_selection import train_test_split
#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'],df['Parti'],random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
#Train and evaluate the model
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)
clfrNB = MultinomialNB(alpha = 0.1)
clfrNB.fit(X_train_vectorized, y_train)
preds = clfrNB.predict(vect.transform(X_test))
score = roc_auc_score(y_test, preds)
confusion_matrix(y_test, preds)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
print("True negatives: " + str(tn))
print("False positives: " + str(fp))
print("False negatives: " + str(fn))
print("True positives: " + str(tp))
print("Probability of detection (sensitivity) " + str(tp/(tp+fn)))
print("True negative rate (specificity) " + str(tn/(tn+fp)))
print(score)

# initial score is an accuracy of 65.6 for 2019, 64.6 for 2011 - 2019

'''
-------------------- Multilabel Classification -------------------
'''

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report


le = preprocessing.LabelEncoder()
target = le.fit_transform(df['Parti'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

vectorizer = TfidfVectorizer(sublinear_tf=True, 
                        max_df=0.3,
                        min_df=100,
                        lowercase=True,
                        stop_words=None, 
                        max_features=20000,
                        tokenizer=None,
                        ngram_range=(1,4)
                        )

train = vectorizer.fit_transform(df['text'])
X_train, X_test, y_train , y_test = train_test_split(train, target, test_size=5000, random_state=0)
clf = MultinomialNB(alpha=.1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(clf.score(X_test, y_test))
print(accuracy_score(y_test, pred))
matrix = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred, target_names=le_name_mapping))


# Accuracy: 34.9 % on 2019 data, 41.6 on 2011-2019 data, 40.5 on 2000-2019



'''
-------------------- Word clouds -------------------
'''

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Create separate dataframes for socialist and liberal parties
liberal = df[(df['Parti'] == 'Venstre') | (df['Parti']=='Det_Konservative_Folkeparti')| (df['Parti']=='Liberal_Alliance')| (df['Parti']=='Dansk_Folkeparti')| (df['Parti']=='Radikale_Venstre')]
liberal = liberal.dropna(subset=['text', 'Parti'])

socialist = df[(df['Parti'] == 'Socialdemokratiet') | (df['Parti']=='Socialistisk_Folkeparti')| (df['Parti']=='Enhedslisten')| (df['Parti']=='Alternativet')]
socialist = socialist.dropna(subset=['text', 'Parti'])


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Liberal word cloud
text = liberal.text.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'white',
    colormap= "Blues",
    stopwords = None).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# Socialist word cloud
text = socialist.text.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'white',
    colormap= "Reds",
    stopwords = None).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()




'''
-------------------- Baseline -------------------
'''


from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
DummyClassifier(strategy='most_frequent')
dummy_clf.predict(X_train)
dummy_clf.score(X_test, y_test)

# Most frequent: 19.06 %
# Stratified: 14.7 %
# Prior: 19.06
# Uniform: 11.32










