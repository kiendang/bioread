from gensim.models.keyedvectors import KeyedVectors

from pathlib import Path


W2V_DIR = Path.home()/'data'


model = KeyedVectors.load_word2vec_format(W2V_DIR/'pubmed2018_w2v_200D'/'pubmed2018_w2v_200D.bin', binary=True)
model.save_word2vec_format(W2V_DIR/'pubmed2018_w2v_200D'/'pubmed2018_w2v_200D.txt', binary=False)
