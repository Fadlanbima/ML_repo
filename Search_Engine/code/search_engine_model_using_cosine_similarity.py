# -*- coding: utf-8 -*-
"""Search Engine model using cosine similarity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hZIjPCalqvHql2HV0vO2Q1fR57uaDaXe

## **Install and Import Module**
"""

!pip3 freeze > requirements.txt

!pip install Sastrawi
!pip install swifter
!pip install unidecode textblob sastrawi
!pip install spacy python-crfsuite unidecode textblob sastrawi

import numpy as np
import pandas as pd
import os
from  IPython import display

import pathlib
import shutil
import tempfile

!pip install -q git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.python.saved_model import saved_model

import pydot
import string
import re
import nltk
import Sastrawi
import pandas as pd
import swifter
import numpy as np
import tensorflow as tf

from nltk.tokenize import TweetTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import defaultdict
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import linear_kernel

nltk.download('popular')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

"""# **Clone Data From Github**"""

!git clone https://github.com/Fadlanbima/ML_repo.git

data = pd.read_csv('/content/ML_repo/data/Data Destinasi Wisatai.csv')

"""# **Some PreProcessing**"""

#casefolding
data['Detail_list'] = data['Detail'].str.lower()
data['jenis_wisata'] = data['Preferensi 2'].str.lower()

#memunculkan preferensi 2 sebanyak 5 kali untuk manambah bobot
data['jenis_wisata'] = data['jenis_wisata'].apply(lambda x: [x,x,x,x,x])

# mengubah list menjadi string
data['jenis_wisata'] = [' '.join(map(str, k)) for k in data['jenis_wisata']]

# menggabungkan data
data['data_ready'] = data['Detail_list']+ " " +data['jenis_wisata']+ " " +data['Destinasi']

#tokenizing
Tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

def tokenization(text):
    teks= Tokenizer.tokenize(text)
    return teks

data['tokenized'] = data['data_ready'].apply(lambda x: tokenization(x))

factory = StopWordRemoverFactory()

Sastrawi_StopWords_id = factory.get_stop_words()

stopword = Sastrawi_StopWords_id + ['rt']

print(stopword)

#remove stopword
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
data['stopword_remove'] = data['tokenized'].apply(lambda x: remove_stopwords(x))

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
def wordLemmatizer(data):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    file_clean_k =pd.DataFrame()
    for index,entry in enumerate(data):
        
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if len(word)>1 and word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
                file_clean_k.loc[index,'Keyword_final'] = str(Final_words)
                file_clean_k.loc[index,'Keyword_final'] = str(Final_words)
                #file_clean_k=file_clean_k.replace(to_replace ="\[.", value = '', regex = True)
                #file_clean_k=file_clean_k.replace(to_replace ="'", value = '', regex = True)
                #file_clean_k=file_clean_k.replace(to_replace =" ", value = '', regex = True)
                #file_clean_k=file_clean_k.replace(to_replace ='\]', value = '', regex = True)
    return file_clean_k

df_clean = wordLemmatizer(data['stopword_remove']) 
df_clean

df_clean=df_clean.replace(to_replace ="\[.", value = '', regex = True)
df_clean=df_clean.replace(to_replace ="'", value = '', regex = True)
df_clean=df_clean.replace(to_replace =" ", value = '', regex = True)
df_clean=df_clean.replace(to_replace ='\]', value = '', regex = True)

data.insert(loc=3, column='Final_Keyword', value=df_clean['Keyword_final'].tolist())

data

data.rename(columns={'Preferensi 2':'preferensi'}, inplace=True)

"""# **Split Data to Training, Testing and Validating**"""

train_data= data
#train_data.dropna(axis = 0, how ='any',inplace=True) 
train_data['Num_words_text'] = train_data['data_ready'].apply(lambda x:len(str(x).split())) 
mask = train_data['Num_words_text'] >2
train_data = train_data[mask]
print('===========Train Data =========')
print(train_data['preferensi'].value_counts())
print(len(train_data))
print('==============================')


test_data= data
#test_data.dropna(axis = 0, how ='any',inplace=True) 
test_data['Num_words_text'] = test_data['Detail'].apply(lambda x:len(str(x).split())) 
mask = test_data['Num_words_text'] >2
test_data = test_data[mask]
print('===========Test Data =========')
print(test_data['preferensi'].value_counts())
print(len(test_data))
print('==============================')

X_train, X_valid, y_train, y_valid = train_test_split(train_data['data_ready'].tolist(), train_data['preferensi'].tolist(), test_size=0.20,stratify = train_data['preferensi'].tolist(), random_state=0)


print('Train data len:'+str(len(X_train)))
print('Class distribution: '+str(Counter(y_train)))
print('Valid data len:'+str(len(X_valid)))
print('Class distribution: '+ str(Counter(y_valid)))




x_train=np.asarray(X_train)
x_valid = np.array(X_valid)
x_test =np.asarray(test_data['Detail'].tolist())

le = LabelEncoder()

train_labels = le.fit_transform(y_train)
train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))

valid_labels = le.transform(y_valid)
valid_labels = np.asarray( tf.keras.utils.to_categorical(valid_labels))

test_labels = le.transform(test_data['preferensi'].tolist())
test_labels = np.asarray(tf.keras.utils.to_categorical(test_labels))
list(le.classes_)


train_ds = tf.data.Dataset.from_tensor_slices((x_train,train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((x_valid,valid_labels))
test_ds = tf.data.Dataset.from_tensor_slices((x_test,test_labels))

print(y_train[:10])
train_labels = le.fit_transform(y_train)
print('Text to number')
print(train_labels[:10])
train_labels = np.asarray( tf.keras.utils.to_categorical(train_labels))
print('Number to category')
print(train_labels[:10])

count =0
print('======Train dataset ====')
for value,label in train_ds:
    count += 1
    print(value,label)
    if count==5:
        break
count =0
print('======Validation dataset ====')
for value,label in valid_ds:
    count += 1
    print(value,label)
    if count==5:
        break
print('======Test dataset ====')
for value,label in test_ds:
    count += 1
    print(value,label)
    if count==5:
        break

"""# **Build Model semantic document search engine**"""

embedding = "https://tfhub.dev/google/nnlm-id-dim50-with-normalization/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)

print(x_train[:1])
hub_layer(x_train[:1])

model = tf.keras.Sequential([hub_layer,
                             tf.keras.layers.Dense(16, activation='relu'),
                             tf.keras.layers.Dropout(0.1),
                             tf.keras.layers.Dense(8, activation='relu'),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(4,activation='sigmoid')
])


model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=["CategoricalAccuracy"])

epochs = 25

# Fit the model using the train and test datasets.
#history = model.fit(x_train, train_labels,validation_data= (x_test,test_labels),epochs=epochs )
history = model.fit(train_ds.shuffle(33).batch(10),
                    epochs= epochs ,
                    validation_data=valid_ds.batch(10),
                    verbose=1)

"""# **Evaluating the Model**"""

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test,test_labels)
print("test loss, test acc:", results)


# Generate predictions (probabilities -- the output of the last layer)
# on test  data using `predict`
print("Generate predictions for all samples")
predictions = model.predict(x_test)
print(predictions)
predict_results = predictions.argmax(axis=1)

"""# **Saved Model**"""

MODEL = "saved_model"

model.save(MODEL)

"""# **Load the Model**"""

# Commented out IPython magic to ensure Python compatibility.
module_url = '/content/rps_saved'
#module_path ="/content/zettadevs/GoogleUSEModel/USE_4"
# %time model = tf.saved_model.load(module_url)
#print ("module %s loaded" % module_url)

#Create function for using modeltraining
def embed(input):
    return model(input)

Model_USE = embed(data.Destinasi[0:2500])

exported = tf.train.Checkpoint(v=tf.Variable(Model_USE))
exported.f = tf.function(
    lambda  x: exported.v * x,
    input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
tf.saved_model.save(exported,'/content/modelgess/')

imported = tf.saved_model.load('/content/modelgess/')
loadedmodel = imported.v.numpy()

ls =[]
chunksize = 66
le =len(data.data_ready)
for i in range(0,le,chunksize):
    if(i+chunksize > le): 
        chunksize= le;
        ls.append(chunksize)
    else:
        a =i+chunksize
        ls.append(a)
ls
j=0
for i in ls:
    directory = "/content/modelgess/" + str(i)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = "/content/modelgess/" + str(i)
    print(j,i) 
    m=embed(data.data_ready[j:i])
    exported_m = tf.train.Checkpoint(v=tf.Variable(m))
    exported_m.f = tf.function(
    lambda  x: exported_m.v * x,
    input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])

    tf.saved_model.save(exported_m,directory)
    j = i
    print(i)

ar =[]
for i in ls:
    directory = "/content/modelgess/" + str(i)
    if os.path.exists(directory):
        print(directory)
        imported_m = tf.saved_model.load(directory)
        a= imported_m.v.numpy()
        #print(a)
        exec(f'load{i} = a')

con_a = np.concatenate((load66,load132,load198,load264,load330,load396,load462,load528,load594,load660))
con_a.shape

imported = tf.saved_model.load('/content/modelgess/')
loadedmodel =imported.v.numpy()
loadedmodel.shape

"""# **Search engine for recomendation**"""

def recommendation(query):
    q = [query]
    # embed the query for calcluating the similarity
    Q_Train = embed(q)
    
    #imported_m = tf.saved_model.load('/content/rps_saved')
    #loadedmodel =imported_m.v.numpy()
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

recommendation('kota')

"""# **Download model to zip, for archive**"""

!zip -r /content/model.zip /content/modelgess

from google.colab import files
files.download("/content/rps_saved.zip")

"""# **Convert Model to TFLite**"""

# Convert model

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir='/content/rps_saved', signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the quantized model to file to the Downloads directory
f = open('docsearch_model.tflite', "wb")
f.write(tflite_model)
f.close()

# Download the digit classification model
from google.colab import files
files.download('docsearch_model.tflite')

print('`docsearch_model.tflite` has been downloaded')
