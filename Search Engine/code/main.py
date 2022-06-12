
import os
import numpy as np
import pandas as pd


# from nltk.tokenize import TweetTokenizer

from sklearn.metrics.pairwise import linear_kernel


import tensorflow as tf
# import tensorflow_hub as hub


import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Content Base Filtering')
	parser.add_argument('--text', type=str, help='input your text')
	args = parser.parse_args()
	return args

args = parse_args()

MODEL_URL = "/content/rps_saved"

DATA_URL = '/content/ML_repo/data/Data Destinasi Wisatai.csv'

MODEL_DIR =  '/content/modelgess/'


def load_data():
    data = pd.read_csv(DATA_URL)
    data = data[['Destinasi', 'Biaya Masuk', 'Jam Operasional']]
    return data

def load_main_model():
    model = tf.saved_model.load(MODEL_URL)

def load_models():
    loaded_models = []

    listdir = os.listdir(MODEL_DIR)

    for folder_name in listdir:
        directory = MODEL_DIR + str(folder_name)

        if os.path.exists(directory):
            imported_m = tf.saved_model.load(directory)
            loaded_model = imported_m.v.numpy()
            loaded_models.append(loaded_model)
    
    return np.concatenate(tuple(loaded_models))



def SearchDocument(query):    
    q = [query]

    model = tf.saved_model.load(MODEL_URL)
    
    Q_Train = model(q)
    con_a = load_models()
    
    data = load_data()

    # Calculate the Similarity
    linear_similarities = linear_kernel(Q_Train, con_a).flatten() 
    #Sort top 10 index with similarity score
    Top_index_doc = linear_similarities.argsort()[:-50:-1]
    # sort by similarity score
    linear_similarities.sort()

    a = pd.DataFrame()
    for i,index in enumerate(Top_index_doc):
        a.loc[i,'index'] = str(index)
        a.loc[i,'destinasi_rekomendasi'] = data['Destinasi'][index] ## Read File name with index from File_data DF
        a.loc[i,'biaya_masuk'] = data['Biaya Masuk'][index]
        a.loc[i,'jam_operasional'] = data['Jam Operasional'][index]
    for j,simScore in enumerate(linear_similarities[:-50:-1]):
        a.loc[j,'Score'] = simScore
    return a

def main():
    from fastapi import FastAPI
    app = FastAPI()
    #app.get("/")
    async def home():
	      return {'message' = 'welcome'}
    output = SearchDocument(args.text)
    print(output)

if __name__ == "__main__":
	main()
