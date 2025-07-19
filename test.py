from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import argparse
from keras.layers import Input,Dense, LSTM, Embedding, Dropout, add
from keras.models import Model

ap=argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True, help="Image")
args=vars(ap.parse_args())
img_path=args['image']

def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer,tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            print("word:",word)
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred_index = np.argmax(pred)
        word = word_for_id(pred_index, tokenizer)
        
        print(f"Step {i+1}:")
        print(f"Input sequence: {in_text}")
        print(f"Predicted word: {word}")
        
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def define_model(vocab_size, max_length):
    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM sequence model
    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())

    return model

max_length=32
tokenizer=load(open("tokenizer.p","rb"))
vocab_size=len(tokenizer.word_index)+1

print(f"Tokenizer vocabulary size: {len(tokenizer.word_index)}")
print(f"Word index sample: {list(tokenizer.word_index.items())[:50]}")


model=define_model(vocab_size, max_length)
model.load_weights("/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/models2/model_5.h5")
xception_model=Xception(include_top=False, pooling="avg")

photo=extract_features(img_path,xception_model)
img=Image.open(img_path)

description=generate_desc(model,tokenizer,photo,max_length)
print(description)