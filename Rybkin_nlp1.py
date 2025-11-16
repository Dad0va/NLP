import gensim
import numpy as np
from gensim.models import KeyedVectors

uk_emb = KeyedVectors.load_word2vec_format("E:/cc.uk.300.vec")
ru_emb = KeyedVectors.load_word2vec_format("E:/cc.ru.300.vec")
ru_emb.most_similar([ru_emb["август"]], topn=10)

uk_emb.most_similar([uk_emb["серпень"]])
ru_emb.most_similar([uk_emb["серпень"]])

def load_word_pairs(filename):
    uk_ru_pairs = []
    uk_vectors = []
    ru_vectors = []
    with open(filename, "r", encoding='utf_8_sig') as inpf:
        for line in inpf:
            uk, ru = line.rstrip().split("\t")
            if uk not in uk_emb or ru not in ru_emb:
                continue
            uk_ru_pairs.append((uk, ru))
            uk_vectors.append(uk_emb[uk])
            ru_vectors.append(ru_emb[ru])
    return uk_ru_pairs, np.array(uk_vectors), np.array(ru_vectors)

uk_ru_train, X_train, Y_train = load_word_pairs("E:/ukr_rus.train.txt")
uk_ru_test, X_test, Y_test = load_word_pairs("E:/ukr_rus.test.txt")

from sklearn.linear_model import LinearRegression
# YOUR CODE HERE
# Example: mapping = LinearRegression(fit_intercept=False).fit(X_train, Y_train)
mapping = LinearRegression().fit(X_train, Y_train)

august = mapping.predict(uk_emb["серпень"].reshape(1, -1))
ru_emb.most_similar(august)

def precision(pairs, mapped_vectors, topn=1):
    """
    :args:
        pairs = list of right word pairs [(uk_word_0, ru_word_0), ...]
        mapped_vectors = list of embeddings after mapping from source embedding space to destination embedding space
        topn = the number of nearest neighbours in destination embedding space to choose from
    :returns:
        precision_val, float number, total number of words for those we can find right translation at top K.
    """
    assert len(pairs) == len(mapped_vectors)
    num_matches = 0
    for i, (_, ru) in enumerate(pairs):
        # YOUR CODE HERE
        for (a, b) in ru_emb.most_similar(mapped_vectors[i].reshape(1, -1), topn=topn):
            if ru == a:
                num_matches += 1
    precision_val = num_matches / len(pairs)
    print(precision_val)
    return precision_val

assert precision([("серпень", "август")], august, topn=5) == 0.0
assert precision([("серпень", "август")], august, topn=9) == 1.0
assert precision([("серпень", "август")], august, topn=10) == 1.0

assert precision(uk_ru_test, Y_test) == 1.0

precision_top1 = precision(uk_ru_test, mapping.predict(X_test), 1)
precision_top5 = precision(uk_ru_test, mapping.predict(X_test), 5)

assert precision_top1 >= 0.635
assert precision_top5 >= 0.811

from numpy.linalg import svd

def learn_transform(X_train, Y_train):
    """
    :returns: W* : float matrix[emb_dim x emb_dim] as defined in formulae above
    """
    # YOU CODE HERE
    c, b, d = svd(np.matmul(X_train.T, Y_train), full_matrices=True)
    return np.matmul(c, d)


W = learn_transform(X_train, Y_train)
ru_emb.most_similar([np.matmul(uk_emb["серпень"], W)])[0]

assert precision(uk_ru_test, np.matmul(X_test, W)) >= 0.653
assert precision(uk_ru_test, np.matmul(X_test, W), 5) >= 0.824

with open("E:/fairy_tale.txt", "r", encoding='utf8') as inpf:
    uk_sentences = [line.rstrip().lower() for line in inpf]

from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

def translate(sentence):
    """
    :args:
        sentence - sentence in Ukrainian (str)
    :returns:
        translation - sentence in Russian (str)

    * find Ukrainian embedding for each word in sentence
    * transform Ukrainian embedding vector
    * find nearest Russian word and replace
    """
    # YOUR CODE HERE
    tokens = tokenizer.tokenize(sentence.lower())
    translated_tokens = []

    for tok in tokens:

        if tok not in uk_emb:
            translated_tokens.append(tok)
            continue

        uk_vec = uk_emb[tok]
        mapped_vec = np.matmul(uk_vec, W)
        ru_word = ru_emb.most_similar([mapped_vec], topn=1)[0][0]
        translated_tokens.append(ru_word)

    return " ".join(translated_tokens)

assert translate(".") == "."
assert translate("1 , 3") == "1 , 3"
assert translate("кіт зловив мишу") == "кот поймал мышку"

translate("кіт зловив мишу")

for sentence in uk_sentences:
    print("src: {}\ndst: {}\n".format(sentence, translate(sentence)))