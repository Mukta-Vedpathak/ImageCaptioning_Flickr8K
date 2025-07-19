import os
import json
import string
import numpy as np
from tqdm import tqdm
from pickle import dump, load
from PIL import Image
import shutil
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout, Embedding, add, LSTMCell, RNN
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array

tqdm().pandas()

#File Paths
COCO_DIR = "path_to_coco_directory"
IMAGES_DIR = "path_to_images_directory"
ANNOTATION_FILE = "path_to_captions"
WORK_DIR = "directory_to_save_work"
FEATURES_FILE = os.path.join(WORK_DIR, "features.p")
TOKENIZER_FILE = os.path.join(WORK_DIR, "tokenizer.p")
DESCRIPTION_FILE = os.path.join(WORK_DIR, "descriptions.txt")

def load_coco_captions(json_path, img_dir):
    with open(json_path, 'r') as f:
        annotations = json.load(f)['annotations']

    data = []
    for ann in annotations:
        img_name = '%012d.jpg' % ann['image_id']
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            caption = ann['caption']
            data.append((img_name, caption))
    descriptions = {}
    for img, caption in data:
        descriptions.setdefault(img, []).append(caption)
    return descriptions

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in descriptions.items():
        for i, cap in enumerate(caps):
            cap = cap.replace("-", " ")
            desc = cap.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1 and word.isalpha()]
            descriptions[img][i] = ' '.join(desc)
    return descriptions

def save_descriptions(descriptions, filename):
    lines = []
    for img, caps in descriptions.items():
        for cap in caps:
            lines.append(f"{img}\t{cap}")
    with open(filename, "w") as f:
        f.write("\n".join(lines))

def load_saved_descriptions(filename):
    descriptions = {}
    with open(filename, 'r') as f:
        for line in f:
            img, cap = line.strip().split('\t')
            descriptions.setdefault(img, []).append(cap)
    return descriptions

print("\n loading captions...")
descriptions = load_coco_captions(ANNOTATION_FILE, IMAGES_DIR)

print("\n cleaning captions...")
descriptions = clean_descriptions(descriptions)

print("\n saving descriptions...")
save_descriptions(descriptions, DESCRIPTION_FILE)

loaded_descriptions = load_saved_descriptions(DESCRIPTION_FILE)

print(f"Total images with captions: {len(loaded_descriptions)}")
first_img = next(iter(loaded_descriptions))
print(f"\nSample image: {first_img}")
print("Captions:", loaded_descriptions[first_img])

def text_vocabulary(descriptions):
    vocab = set()
    for descs in descriptions.values():
        for d in descs:
            vocab.update(d.split())
    return vocab

def dict_to_list(descriptions):
    return [f"<start> {d} <end>" for descs in descriptions.values() for d in descs]

def create_tokenizer(descriptions):
    lines = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

print("\n creating tokenizer...")
tokenizer = create_tokenizer(descriptions)
dump(tokenizer, open(TOKENIZER_FILE, "wb"))
print("\n saved tokenizer...")
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocab_size: {vocab_size}")

def get_max_length(descriptions):
    lines = dict_to_list(descriptions)
    return max(len(d.split()) for d in lines)

max_length = get_max_length(descriptions)
print(f"✅ Max Caption Length: {max_length}")

def extract_features(directory):
    model = Xception(include_top=False, pooling="avg")
    features = {}
    for img in tqdm(os.listdir(directory)):
        if not img.endswith(".jpg"):
            continue
        path = os.path.join(directory, img)
        try:
            image = Image.open(path).resize((299, 299)).convert("RGB")
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            feature = model.predict(image, verbose=0)
            features[img] = feature
        except Exception as e:
            print(f"Failed: {img} - {e}")
    dump(features, open(FEATURES_FILE, "wb"))
    return features

print("\n extracting features...")
features = extract_features(IMAGES_DIR)

print("\n loading features...")
all_features = load(open(FEATURES_FILE, "rb"))
print("\n training descriptions...")
train_descriptions = {k: v for k, v in loaded_descriptions.items() if k in all_features}
print("\n training features...")
train_features = {k: all_features[k] for k in train_descriptions}


def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([f"<start> {desc} <end>"])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, features, tokenizer, max_length):
    def generator():
        for key, descs in descriptions.items():
            feature = features[key][0]
            input_img, input_seq, output_word = create_sequences(tokenizer, max_length, descs, feature)
            for i in range(len(input_img)):
                yield {'input_1': input_img[i], 'input_2': input_seq[i]}, output_word[i]
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature).batch(32).prefetch(tf.data.AUTOTUNE)

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    lstm = RNN(LSTMCell(256))(se2)

    decoder1 = add([fe2, lstm])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

def get_steps(descriptions):
    total = sum(len(d.split()) - 1 for descs in descriptions.values() for d in descs)
    return max(1, total // 32)

model = define_model(vocab_size, max_length)
steps = get_steps(train_descriptions)

models_dir = os.path.join(WORK_DIR, "models")
if os.path.exists(models_dir):
    shutil.rmtree(models_dir)
os.mkdir(models_dir)

for i in range(10):
    dataset = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit(dataset, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save(f"{models_dir}/model_{i}.h5")