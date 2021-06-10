from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm, naive_bayes
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
import gensim
from gensim.models import Word2Vec
import numpy as np

warnings.filterwarnings('ignore')  # Warning people, manual storage

pos_revs_path = 'D:\\python-projects\\NLP-ASSIGNMENT2\\txt_sentoken\\pos'  # change this to your local path
neg_revs_path = 'D:\\python-projects\\NLP-ASSIGNMENT2\\txt_sentoken\\neg'  # change this to your local path


def load_data(folder_path):
    files = os.listdir(folder_path)
    reviews = []
    for file in files:
        with open(folder_path + '\\' + file) as f:
            reviews.append(f.read())
    return reviews


def word2vec():
    neg_revs = load_data('D:\\python-projects\\NLP-ASSIGNMENT2\\txt_sentoken\\neg')
    pos_revs = load_data('D:\\python-projects\\NLP-ASSIGNMENT2\\txt_sentoken\\pos')
    reviews = pos_revs + neg_revs
    preprocessed_reviews = [gensim.utils.simple_preprocess(review) for review in reviews]
    # print(preprocessed_reviews)
    # model = gensim.models.Word2Vec(sentences=preprocessed_reviews, vector_size=100, window=5, workers=4)
    # model.save('model.model')
    model = Word2Vec.load('model.model')
    # feature_vectors = model.wv.vectors
    mean_vectors = []
    for rev in preprocessed_reviews:
        words = calc_mean(rev, model)
        mean_vectors.append(np.mean(model.wv[words], axis=0))
    # print(len(mean_vectors))
    return mean_vectors


def calc_mean(rev, model):
    words_in_word2vec = []
    for word in rev:
        if word in model.wv.index_to_key:
            words_in_word2vec.append(word)
    return words_in_word2vec


def create_model():
    labels = []
    for i in range(0, 2000):
        if i < 1000:
            labels.append(1)
        else:
            labels.append(0)
    data = word2vec()
    # print(len(data))
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    # model = LogisticRegression(random_state=0).fit(x_train, y_train)
    model = LinearRegression()
    model.fit(x_train, y_train)
    joblib.dump(model, 'model.pkl')
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    return train_acc, test_acc


# word2vec()
print(create_model())

# linear regression CBW (0.3768796383506482, 0.2271725424390647)
# logistic regression CBOW (0.703125, 0.6825)
# svm CBOW (0.675625, 0.6575)
# naive_bias GaussianNB CBOW (0.611875, 0.605)
