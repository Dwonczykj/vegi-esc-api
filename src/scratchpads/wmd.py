from __future__ import print_function
from typing import Any

# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')


def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def compare(model, new_sentence:str):
    # ~ https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html
    sentence_orange = preprocess(new_sentence)
    sentence_obama = 'Obama speaks to the media in Illinois'
    sentence_president = 'The president greets the press in Chicago'
    distance = model.wmdistance(sentence_obama, sentence_orange)
    print('distance = %.4f' % distance)
    return 'distance = %.4f' % distance

def example(model):
    return compare(model,'Oranges are my favorite fruit')

if __name__ == '__main__':
    import gensim.downloader as api
    model:Any = api.load('word2vec-google-news-300')
    sentence_orange = preprocess('Oranges are my favorite fruit')
    sentence_obama = 'Obama speaks to the media in Illinois'
    distance = model.wmdistance(sentence_obama, sentence_orange)

