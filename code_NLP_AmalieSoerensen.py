from path import data_path
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import balanced_accuracy_score, classification_report, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from afinn import Afinn
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import statistics
import seaborn as sns


df = pd.read_csv(data_path)

'''
-------------------- Preprocessing -------------------
'''

# Keep only speeches from 2015 and later
df['Dato'] = df['Dato'].str.slice(start = 6, stop = 10)
df['Dato'] = df['Dato'].astype(int)
df = df[df['Dato'] > 2014]

# Unique speakers
speakers = df['Title']
unique_speakers = df['Title'].unique()

# Filter out formanden
df = df[~df.Title.isin(['formand', 'Henrik Dam Kristensen', 'Leif Mikkelsen', 'Næstformand', 'Pia Kjærsgaard', 'næstformand'])]

# Replace nan, party names, etc.
df['text'] = df.text.str.replace('nan' , '')
df['text'] = df.text.str.replace('tikke' , 'tak')
df['text'] = df.text.str.replace('matiasse' , 'mathias')
df['text'] = df.text.str.replace('alternativ' , '')
df['text'] = df.text.str.replace('dansk folkeparti' , '')
df['text'] = df.text.str.replace('liberal alliance' , '')
df['text'] = df.text.str.replace('radikal venstre' , '')
df['text'] = df.text.str.replace('radikal' , '')
df['text'] = df.text.str.replace('socialdemokrati' , '')
df['text'] = df.text.str.replace('socialistisk folkeparti' , '')
df['text'] = df.text.str.replace('det konservativ folkeparti' , '')
df['text'] = df.text.str.replace('venstre' , '')
df['text'] = df.text.str.replace('enhedslisten' , '')
df['text'] = df.text.str.replace('enhedslist' , '')
df['text'] = df.text.str.replace('konservativ folkeparti' , '')
df['text'] = df.text.str.replace('sk' , '')

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
df['Parti'] = df['Parti'].replace({'Konservative': 'Det_Konservative_Folkeparti'})

df = df.loc[df['Parti'].isin(['Dansk_Folkeparti', 'Radikale_Venstre', 'Venstre', 'Socialdemokratiet', 'Socialistisk_Folkeparti', 'Liberal_Alliance', 'Det_Konservative_Folkeparti', 'Enhedslisten', 'Alternativet'])]

# Remove NaN from text and Parti and check that it worked
df = df.dropna(subset=['text', 'Parti'])
df['text'].isna().values.sum()


'''
-------------------- Binary Classification -------------------
'''

df['Parti_b'] = df['Parti'].copy()

# Binary Classification liberal/socialist
df['Parti_b'] = df['Parti_b'].replace({'Det_Konservative_Folkeparti': 0})
df['Parti_b'] = df['Parti_b'].replace({'Dansk_Folkeparti': 0})
df['Parti_b'] = df['Parti_b'].replace({'Radikale_Venstre': 0})
df['Parti_b'] = df['Parti_b'].replace({'Venstre': 0})
df['Parti_b'] = df['Parti_b'].replace({'Socialdemokratiet': 1})
df['Parti_b'] = df['Parti_b'].replace({'Socialistisk_Folkeparti': 1})
df['Parti_b'] = df['Parti_b'].replace({'Liberal_Alliance': 0})
df['Parti_b'] = df['Parti_b'].replace({'Enhedslisten': 1})
df['Parti_b'] = df['Parti_b'].replace({'Alternativet': 1})

# TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer(sublinear_tf=True, 
                        max_df=0.3,
                        min_df=100,
                        lowercase=True,
                        stop_words=None, 
                        max_features=20000,
                        tokenizer=None,
                        ngram_range=(1,4)
                        )

# Count vectorizer
vectorizer_count = CountVectorizer(ngram_range =(1,4), max_features=20000)

# Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(df['text'],df['Parti_b'],random_state=0)

# Vectorize train, count
vect_count = vectorizer_count.fit(X_train)
X_train_vectorized_count = vect_count.transform(X_train)

# Vectorize train, tf-idf
vect_tfidf = vectorizer_tfidf.fit(X_train)
X_train_vectorized_tfidf = vect_tfidf.transform(X_train)

# Fit model, count
clf = MultinomialNB(alpha = 0.1)
clf.fit(X_train_vectorized_count, y_train)
pred_count = clf.predict(vect_count.transform(X_test))
print(" Count, Balanced accuracy score = " + str(balanced_accuracy_score(y_test, pred_count)))
print(" Count Accuracy score = " + str(accuracy_score(y_test, pred_count)))
print(" Count, Precision = " + str(precision_score(y_test, pred_count)))
print(" Count, Recall = " + str(recall_score(y_test, pred_count)))

# Fit model, tfidf
clf = MultinomialNB(alpha = 0.1)
clf.fit(X_train_vectorized_tfidf, y_train)
pred_tfidf = clf.predict(vect_tfidf.transform(X_test))
print(" Tf-idf, Balanced accuracy score = " + str(balanced_accuracy_score(y_test, pred_tfidf)))
print(" Tf-idf, Accuracy score = " + str(accuracy_score(y_test, pred_tfidf)))
print(" Tf-idf, Precision = " + str(precision_score(y_test, pred_tfidf)))
print(" Tf-idf, Recall = " + str(recall_score(y_test, pred_tfidf)))

# Binary, count, acurracy: 69.3, balanced accuracy: 69.3, precision: 68.5, recall 71.7
# Binary, tdidf, acurracy: 65.5, balanced accuracy: 65.5, precision 65.2, recall 67.1

'''
-------------------- Multilabel Classification -------------------
'''

# Give each party a number
le = preprocessing.LabelEncoder()
target = le.fit_transform(df['Parti'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# Split data in train and test
X_train, X_test, y_train, y_test = train_test_split(df['text'],df['Parti'],random_state=0)

# Vectorize train, count
vect_count = vectorizer_count.fit(X_train)
X_train_vectorized_count = vect_count.transform(X_train)

# Vectorize train, tf-idf
vect_tfidf = vectorizer_tfidf.fit(X_train)
X_train_vectorized_tfidf = vect_tfidf.transform(X_train)

# Fit model, count
clf = LogisticRegression(random_state=0, class_weight = 'balanced', max_iter=1000).fit(X_train_vectorized_count, y_train)
pred_count_balanced = clf.predict(vect_count.transform(X_test))
print(" Count, Balanced accuracy score = " + str(balanced_accuracy_score(y_test, pred_count_balanced)))
print(" Count Accuracy score = " + str(accuracy_score(y_test, pred_count_balanced)))
report_count = classification_report(y_test, pred_count_balanced)

# Fit model, tfidf
clf = LogisticRegression(random_state=0, class_weight = 'balanced', max_iter=1000).fit(X_train_vectorized_tfidf, y_train)
pred_tfidf_balanced = clf.predict(vect_tfidf.transform(X_test))
print(" Tf-idf, Balanced accuracy score = " + str(balanced_accuracy_score(y_test, pred_tfidf_balanced)))
print(" Tf-idf, Accuracy score = " + str(accuracy_score(y_test, pred_tfidf_balanced)))
report_tfidf = classification_report(y_test, pred_tfidf_balanced)

# count balanced accuracy: 42
# count accuracy: 43.5
# tf-idf balanced accuracy: 43.5
# tf-idf accuracy: 41.5

'''
-------------------- Baseline Models -------------------
'''

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
DummyClassifier(strategy='uniform')
dummy_clf.predict(X_train)
dummy_clf.score(X_test, y_test)

# Multilabel
# Most frequent: 19.7 %
# Stratified: 14.2 %
# Uniform: 11.3

# Binary 50.46 %


'''
------------------- Feature importance -----------------
'''

def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              ", ".join(feature_names[j] for j in top10)))

print_top10(vectorizer_tfidf, clf, le_name_mapping)
print_top10(vectorizer_count, clf, le_name_mapping)


'''
-------------------- Sentiment scores -------------------
'''

# Calculate sentiment scores
afinn = Afinn(language='da')
df['sentiment'] = df['text'].apply(afinn.score)
df['sentiment_category'] = ['positive' if score > 0 
                          else 'negative' if score < 0 
                              else 'neutral' 
                                  for score in df['sentiment']]

# Print mean sentiment score per party
for parti in df.Parti.unique():
    print(parti + ": " + str(statistics.mean(df.sentiment[df.Parti == parti])))

# Plot sentiment scores
df_ny = df.copy()
df_ny['Parti'] = df_ny['Parti'].replace({'Dansk_Folkeparti': 'DF'})
df_ny['Parti'] = df_ny['Parti'].replace({'Radikale_Venstre': 'RV'})
df_ny['Parti'] = df_ny['Parti'].replace({'Socialistisk_Folkeparti': 'SF'})
df_ny['Parti'] = df_ny['Parti'].replace({'Liberal_Alliance': 'LA'})
df_ny['Parti'] = df_ny['Parti'].replace({'Socialdemokratiet': 'S'})
df_ny['Parti'] = df_ny['Parti'].replace({'Det_Konservative_Folkeparti': 'Kons.'})
df_ny['Parti'] = df_ny['Parti'].replace({'Enhedslisten': 'Enh.'})
df_ny['Parti'] = df_ny['Parti'].replace({'Alternativet': 'Alt.'})
df_ny['Parti'] = df_ny['Parti'].replace({'Venstre': 'V'})

palette ={"DF":"steelblue","RV":"deeppink","SF":"salmon", "LA":"orange", "S":"firebrick", "Kons.":"darkgreen", "Enh.":"crimson", "Alt.":"chartreuse", "V":"cornflowerblue"}
sns.set(style="whitegrid")
s = sns.stripplot(x="Parti", y="sentiment", data=df_ny, palette = palette, jitter=True)
fig = s.get_figure()
fig.savefig("sentiment_scores.png")

# Plot sentiment categories for each party
fc = sns.factorplot(x="Parti", hue="sentiment_category", 
                    data=df_ny, kind="count",
                    palette={"negative": "#FE2020", 
                             "positive": "#BADD07", 
                             "neutral": "#68BFF5"})

fc.savefig("sentiment_category.png")

# Create separate dataframes for socialist and liberal parties
liberal = df[(df['Parti'] == 'Venstre') | (df['Parti']=='Det_Konservative_Folkeparti')| (df['Parti']=='Liberal_Alliance')| (df['Parti']=='Dansk_Folkeparti')| (df['Parti']=='Radikale_Venstre')]
liberal = liberal.dropna(subset=['text', 'Parti'])

socialist = df[(df['Parti'] == 'Socialdemokratiet') | (df['Parti']=='Socialistisk_Folkeparti')| (df['Parti']=='Enhedslisten')| (df['Parti']=='Alternativet')]
socialist = socialist.dropna(subset=['text', 'Parti'])

# Mean sentiment for liberal and socialist
statistics.mean(liberal.sentiment)
statistics.mean(socialist.sentiment)


'''
-------------------- Word clouds -------------------
'''

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









