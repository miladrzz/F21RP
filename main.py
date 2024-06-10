import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#import data (.csv file)
spam_file = pd.read_csv('spam.csv')
spam_file['spam'] = spam_file['Category'].apply(lambda x: 1 if x=='spam' else 0)
x_train, x_test, y_train, y_test = train_test_split(spam_file.Message,spam_file.spam)

cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
x_train_count.toarray()

model = MultinomialNB()
model.fit(x_train_count, y_train)

new_message_ham=["Hey bro, wanna work on the project?"]
new_message_ham_count = cv.transform(new_message_ham)

new_message_spam = ["click to collect your reward!!"]
new_message_spam_count = cv.transform(new_message_spam)

x_test_count = cv.transform(x_test)
print(model.score(x_test_count,y_test))