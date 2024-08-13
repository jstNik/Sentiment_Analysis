import os
import re
import string

import contractions
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def cleaning(text):

    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english', True)

    # converting to lowercase, removing URL links, special characters, punctuations...
    text = text.lower()  # converting to lowercase
    text = contractions.fix(text)
    text = re.sub('https?://\S+|www\.\S+', '', text)  # removing URL links
    text = re.sub(r"\b\d+\b", '', text)  # removing number
    text = re.sub('<.*?>+', '', text)  # removing special characters,
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # punctuations
    text = re.sub('\n', '', text)
    text = re.sub('[’“”…]', '', text)

    # removing emoji:
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    def stem_lem(txt):
        txt = lemmatizer.lemmatize(txt)
        txt = stemmer.stem(txt)
        return txt

    text = ' '.join(list(map(lemmatizer.lemmatize, word_tokenize(text))))
    text = text.replace('not', '')

    return text


def main():
    directories = ['dataset/tokens/neg/', 'dataset/tokens/pos/']
    data = []

    for i in range(len(directories)):
        for path in os.listdir(directories[i]):
            file = open(directories[i] + path)
            review = cleaning(file.read())
            data.append([i, review])
            file.close()

    df = pd.DataFrame(data, columns=['sentiment', 'review'])

    unigram_cv = CountVectorizer(stop_words='english', ngram_range=(1, 1))
    txt = unigram_cv.fit_transform(df['review'])
    x_train, x_test, y_train, y_test = train_test_split(txt, df['review'], test_size=0.2, shuffle=True)

    # Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    predicted = nb.predict(x_test)
    accuracy = accuracy_score(predicted, y_test)
    print(accuracy)

    # Random Forest
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    predicted = rfc.predict(x_test)
    accuracy = accuracy_score(predicted, y_test)
    print(accuracy)


    pass


if __name__ == "__main__":
    main()
