import keras
import tensorflow as tf
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import pickle
from tqdm import tqdm
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten
from keras.layers import merge
from keras.optimizers  import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import nltk
import os

token = '../input/img-caption-data/Flickr8k_text/Flickr8k.token.txt'
captions = open(token, 'r').read().strip().split('\n')


d = {}
for i, row in enumerate(captions):
    row = row.split('\t')
    row[0] = row[0][:len(row[0])-2]
    if row[0] in d:
        d[row[0]].append(row[1])
    else:
        d[row[0]] = [row[1]]
        
d['1000268201_693b08cb0e.jpg']

images = '../input/img-caption-data/Flickr8k_Dataset/Flicker8k_Dataset/'
img = glob.glob(images+'*.jpg')
img[:5]

def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp

train_images_file = '../input/img-caption-data/Flickr8k_text/Flickr_8k.trainImages.txt'
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
train_img = split_data(train_images)
len(train_img)

val_images_file = '../input/img-caption-data/Flickr8k_text/Flickr_8k.devImages.txt'
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))
val_img = split_data(val_images)
len(val_img)

test_images_file = '../input/img-caption-data/Flickr8k_text/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
test_img = split_data(test_images)
len(test_img)

Image.open(train_img[0])


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

plt.imshow(np.squeeze(preprocess(train_img[0])))

model = InceptionV3(weights='imagenet')

from keras.models import Model

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)

tryi = model_new.predict(preprocess(train_img[0]))
tryi.shape

def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

encoding_train = {}
for img in tqdm(train_img):
    encoding_train[img[len(images):]] = encode(img)

with open("encoded_images_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

encoding_train = pickle.load(open('encoded_images_inceptionV3.p', 'rb'))
encoding_train['3556792157_d09d42bef7.jpg'].shape

encoding_test = {}
for img in tqdm(test_img):
    encoding_test[img[len(images):]] = encode(img)

with open("encoded_images_test_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)

encoding_test = pickle.load(open('encoded_images_test_inceptionV3.p', 'rb'))
encoding_test[test_img[0][len(images):]].shape

train_d = {}
for i in train_img:
    if i[len(images):] in d:
        train_d[i] = d[i[len(images):]]

print("Length of train_d: ",len(train_d))

val_d = {}
for i in val_img:
    if i[len(images):] in d:
        val_d[i] = d[i[len(images):]]
        
print("Length of val_d: ",len(val_d))

test_d = {}
for i in test_img:
    if i[len(images):] in d:
        test_d[i] = d[i[len(images):]]
        
print("Length of test_d: ",len(test_d))

train_d[images+'3556792157_d09d42bef7.jpg']


caps = []
for key, val in train_d.items():
    for i in val:
        caps.append('<start> ' + i + ' <end>')

words = [i.split() for i in caps]
unique = []
for i in words:
    unique.extend(i)
    
unique = list(set(unique))

with open("unique.p", "wb") as pickle_d:
    pickle.dump(unique, pickle_d)

unique = pickle.load(open('unique.p', 'rb'))
len(unique)


word2idx = {val:index for index, val in enumerate(unique)}
word2idx['<start>']

idx2word = {index:val for index, val in enumerate(unique)}
idx2word[5553]


max_len = 0
for c in caps:
    c = c.split()
    if len(c) > max_len:
        max_len = len(c)
max_len

len(unique), max_len

vocab_size = len(unique)
vocab_size


f = open('flickr8k_training_dataset.txt', 'w')
f.write("image_id\tcaptions\n")

for key, val in train_d.items():
    for i in val:
        f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")

f.close()

df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
len(df)

c = [i for i in df['captions']]
len(c)

imgs = [i for i in df['image_id']]
a = c[-1]
a, imgs[-1]

for i in a.split():
    print (i, "=>", word2idx[i])

samples_per_epoch = 0
for ca in caps:
    samples_per_epoch += len(ca.split())-1
    
samples_per_epoch



def data_generator(batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        
        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_train[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    n = np.zeros(vocab_size)
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        yield ([images, partial_caps], next_words)
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0


embedding_size = 300


image_model = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        RepeatVector(max_len)
    ])


caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])

type(image_model)

# max_len = 40
image_in = keras.Input(shape=(2048,))
caption_in = keras.Input(shape=(max_len,))


# image_in = Input(shape=(2048,))
# caption_in = keras.Input(shape=(max_len, vocab_size))
caption_in = keras.Input(shape=(max_len,))
merged = keras.layers.concatenate([image_model(image_in), caption_model(caption_in)],axis=1)
latent = Bidirectional(LSTM(256, return_sequences=False))(merged)
out = Dense(vocab_size, activation='softmax')(latent)
final_model = Model([image_in, caption_in], out)

final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

final_model.summary()

steps_per_epoch = samples_per_epoch/128

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=1)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.optimizer.lr = 1e-4
final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=1)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.save_weights('time_inceptionV3_7_loss_3.2604.h5')

final_model.load_weights('time_inceptionV3_7_loss_3.2604.h5')

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.save_weights('time_inceptionV3_3.21_loss.h5')

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.save_weights('time_inceptionV3_3.15_loss.h5')

final_model.fit(data_generator(batch_size=128), steps_per_epoch=samples_per_epoch, epochs=1, verbose=2)

final_model.load_weights('time_inceptionV3_1.5987_loss.h5')


def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encoding_test[image[len(images):]]
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

def beam_search_predictions(image, beam_index = 3):
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encoding_test[image[len(images):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

try_image = test_img[0]
Image.open(try_image)

print ('Normal Max search:', predict_captions(try_image)) 
print ('Beam Search, k=3:', beam_search_predictions(try_image, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(try_image, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(try_image, beam_index=7))



try_image2 = test_img[7]
Image.open(try_image2)

print ('Normal Max search:', predict_captions(try_image2)) 
print ('Beam Search, k=3:', beam_search_predictions(try_image2, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(try_image2, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(try_image2, beam_index=7))



try_image3 = test_img[851]
Image.open(try_image3)

print ('Normal Max search:', predict_captions(try_image3)) 
print ('Beam Search, k=3:', beam_search_predictions(try_image3, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(try_image3, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(try_image3, beam_index=7))



try_image4 = 'Flickr8k_Dataset/Flicker8k_Dataset/136552115_6dc3e7231c.jpg'
print ('Normal Max search:', predict_captions(try_image4))
print ('Beam Search, k=3:', beam_search_predictions(try_image4, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(try_image4, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(try_image4, beam_index=7))
Image.open(try_image4)



im = 'Flickr8k_Dataset/Flicker8k_Dataset/1674612291_7154c5ab61.jpg'
print ('Normal Max search:', predict_captions(im))
print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
Image.open(im)



im = 'Flickr8k_Dataset/Flicker8k_Dataset/384577800_fc325af410.jpg'
print ('Normal Max search:', predict_captions(im))
print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
Image.open(im)



im = 'Flickr8k_Dataset/Flicker8k_Dataset/3631986552_944ea208fc.jpg'
print ('Normal Max search:', predict_captions(im))
print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
Image.open(im)



im = 'Flickr8k_Dataset/Flicker8k_Dataset/3320032226_63390d74a6.jpg'
print ('Normal Max search:', predict_captions(im))
print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
Image.open(im)



im = 'Flickr8k_Dataset/Flicker8k_Dataset/3316725440_9ccd9b5417.jpg'
print ('Normal Max search:', predict_captions(im))
print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
Image.open(im)



im = 'Flickr8k_Dataset/Flicker8k_Dataset/2306674172_dc07c7f847.jpg'
print ('Normal Max search:', predict_captions(im))
print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
Image.open(im)



im = 'Flickr8k_Dataset/Flicker8k_Dataset/2542662402_d781dd7f7c.jpg'
print ('Normal Max search:', predict_captions(im))
print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
Image.open(im)



im = test_img[int(np.random.randint(0, 1000, size=1))]
print (im)
print ('Normal Max search:', predict_captions(im))
print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
Image.open(im)
