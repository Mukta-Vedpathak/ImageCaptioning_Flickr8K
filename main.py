import string
import os
import numpy as np
from pickle import dump,load
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras.applications.xception import Xception,preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, get_file
from keras.layers import add
from keras.models import Model,load_model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from tqdm import tqdm
import time
from PIL import Image
import warnings
import absl.logging
import shutil

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# warnings.filterwarnings("ignore", category=UserWarning)
# absl.logging.set_verbosity(absl.logging.ERROR)

tqdm().pandas()

#Path to captions
dataset_text="/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/Flickr8k_text"

#Path to images
dataset_images="/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/Flicker8k_Dataset"

#Load Doc
def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

# INPUT FILE:
# image1.jpg.1\tA cat sitting on a mat.
# image1.jpg.2\tA cat relaxing indoors.
# image2.jpg.1\tA dog playing in the park.

def all_img_captions(filename):
    file=load_doc(filename)
    captions = file.split('\n')
    descriptions={}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]]=[caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

# DESCRIPTIONS{}
# {
#     'image1.jpg': ['A cat sitting on a mat.', 'A cat relaxing indoors.'],
#     'image2.jpg': ['A dog playing in the park.']
# }

def cleaning_text(captions):
    table=str.maketrans('','', string.punctuation) #removes punctuation
    for img,caps in captions.items():
        for i, img_caption in enumerate(caps):
            img_caption.replace("-","") #remove -
            desc = img_caption.split() # split caption into words
            desc = [word.lower() for word in desc] # convert to lower case
            desc = [word.translate(table) for word in desc] #remove punctuation
            desc = [word for word in desc if (len(word)>1) ] # remove short words
            desc = [word for word in desc if(word.isalpha())] #keep only alphabetic words

            img_caption=' '.join(desc) #rejoin words with space ' ' to form sentence
            captions[img][i]=img_caption # update dictionary descriptions
    return captions

def text_vocabulary(descriptions):
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab #Returns the complete set of unique words extracted from all captions.
 
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
        data = '\n'.join(lines)
        file = open (filename, "w")
        file.write(data)
        file.close()


filename = dataset_text + "/Flickr8k.token.txt"
descriptions = all_img_captions (filename)
#print ("Length of descriptions =", len(descriptions))

clean_descriptions = cleaning_text(descriptions)
vocabulary= text_vocabulary(clean_descriptions)
#print("Length of vocbulary = ", len(vocabulary))

#================================================================================================================
#DESCRIPTIONS
save_descriptions(clean_descriptions, "descriptions.txt")

def download_with_retry(url,filename, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_file(filename, url)
        except Exception as e:
            if attempt == max_retries-1:
                raise e
            print(f"Download attempt failed")
            time.sleep(3)

weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url, 'xceptions_weights,h5')

#Xception Model
model = Xception(include_top=False, pooling="avg", weights="imagenet")

def extract_features(directory):
    features = {}
    valid_images = {'.jpg', '.jpeg', '.png'}
    for img in tqdm(os.listdir(directory)):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            #print(f"Skipping invalid file: {img}")
            continue
        filename = os.path.join(directory, img)
        #print(f"Processing image: {filename}")
        try:
            image = Image.open(filename)
            image = image.resize((299, 299))
            image = np.expand_dims(image, axis=0)
            image = image / 127.5 - 1.0
            feature = model.predict(image)
            features[img] = feature
        except Exception as e:
            print(f"Failed to process image {filename}: {e}")
    #print("Extracted features:", features)
    return features

#extracted features of dataset_images saved in features.p
features = extract_features (dataset_images)
dump(features, open ("features.p","wb"))

#=======================================================================================================

#Load Features File
features = load (open ("features.p", "rb"))

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present=[photo for photo in photos if os.path.exists(os.path.join(dataset_images,photo))]
    return photos_present

def load_clean_descriptions(filename, photos):
    file= load_doc(filename)
    descriptions={}
    for line in file.split("\n"):
        words=line.split()
        if len(words)<1:
            continue
        image, image_caption=words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image]=[]
            desc = '<start> ' + " ".join(image_caption) + ' <end>' # add keywords start and end for the image caption
            descriptions[image].append(desc)
    #print("descriptions", descriptions)
    return descriptions

def load_features(photos):
    all_features = load(open("features.p", "rb"))
    #print("Keys in features.p:", list(all_features.keys()))
    #print("Photos provided:", photos)
    
    # Filter features
    features = {k: all_features[k] for k in photos if k in all_features}
    #print("Matched features:", features)
    return features

#For training dataset
filename = "/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/Flickr8k_text/Flickr_8k.trainImages.txt" #Path to Training Images
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/descriptions.txt", train_imgs) #Path to Descriptions
train_features=  load_features(train_imgs)

#Tokenize training dataset descriptions
def dict_to_list(descriptions):
    all_desc=[]
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    desc_list= dict_to_list(descriptions)
    tokenizer =Tokenizer()
    tokenizer.fit_on_texts(tqdm(desc_list, desc="Fitting tokenizer"))
    return tokenizer


#==============================================================================================================
#TOKENIZER
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open("tokenizer.p", "wb"))

vocab_size=len(tokenizer.word_index)+1
print(vocab_size)


#padding
def max_length(descriptions):
    desc_list= dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

max_length=max_length(train_descriptions)
print(max_length)


#data generator, used by model.fit()
def data_generator(descriptions, features, tokenizer, max_length):
    def generator():
        while True:
            for key, description_list in descriptions.items():
                feature = features[key][0]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
                for i in range(len(input_image)):
                    yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]
    
    # Define the output signature for the generator
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    
    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    return dataset.batch(32)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

#check the shape of the input and output for your model
dataset = data_generator(train_descriptions, features, tokenizer, max_length)
for (a, b) in dataset.take(1):
    print(a['input_1'].shape, a['input_2'].shape, b.shape)
    break

# MODEL
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
    plot_model(model, to_file='model.png', show_shapes=True)

    return model

# train our model
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)

model = define_model(vocab_size, max_length)
epochs = 10

def get_steps_per_epoch(train_descriptions):
    total_sequences = 0
    for img_captions in train_descriptions.values():
        for caption in img_captions:
            words = caption.split()
            total_sequences += len(words) - 1
    return max(1, total_sequences // 32)


steps = get_steps_per_epoch(train_descriptions)

#Save Models
os.mkdir("models")
for i in range(epochs):
    dataset = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit(dataset, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")