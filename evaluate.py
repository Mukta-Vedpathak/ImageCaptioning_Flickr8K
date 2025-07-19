from keras.models import load_model
from keras.applications.xception import Xception
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from pickle import load
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from keras.utils import custom_object_scope
from keras.layers import Layer
from keras.layers import Input,Dense, LSTM, Embedding, Dropout, add
from keras.models import Model
import matplotlib.pyplot as plt

# Paths
dataset_text = "/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/Flickr8k_text"
dataset_images = "/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/Flicker8k_Dataset"

# Utility functions
def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def load_photos(filename):
    file = load_doc(filename)
    photos = file.strip().split('\n')
    return [photo for photo in photos if os.path.exists(os.path.join(dataset_images, photo))]

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.strip().split('\n'):
        words = line.split()
        if len(words) < 1:
            continue
        image_id, image_desc = words[0], words[1:]
        if image_id in photos:
            if image_id not in descriptions:
                descriptions[image_id] = []
            desc = ' '.join(image_desc)
            descriptions[image_id].append(desc)
    return descriptions

def load_features(photos):
    all_features = load(open("features.p", "rb"))
    return {k: all_features[k] for k in photos if k in all_features}

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = [], []
    for key in tqdm(descriptions, desc="Evaluating"):
        y_pred = generate_desc(model, tokenizer, photos[key], max_length)
        references = [d.split() for d in descriptions[key]]
        actual.append(references)
        predicted.append(y_pred.split())
    
    print("\nBLEU Scores:")
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print("BLEU-3: %f" % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print("BLEU-4: %f" % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

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

# Load tokenizer and model
tokenizer = load(open("/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/kaggle_coco_118K/tokenizer_full.p", "rb"))
vocab_size = len(tokenizer.word_index) + 1
max_length = 32

# def evaluate_model(model, descriptions, photos, tokenizer, max_length):
#     actual, predicted = [], []
#     for key in tqdm(descriptions, desc="Evaluating"):
#         y_pred = generate_desc(model, tokenizer, photos[key], max_length)
#         references = [d.split() for d in descriptions[key]]
#         actual.append(references)
#         predicted.append(y_pred.split())

#     b1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
#     b2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
#     b3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
#     b4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

#     return [b1, b2, b3, b4]

# val_imgs = load_photos("/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/Flickr8k_text/Flickr_8k.devImages.txt")
# val_descriptions = load_clean_descriptions("descriptions.txt", val_imgs)
# val_features = load_features(val_imgs)

# val_bleu_scores = []

# for i in range(10):
#     model_path = f"models2/model_{i}.h5"
#     print(f"\nEvaluating model {i} on validation set...")
#     model = define_model(vocab_size, max_length)
#     model.load_weights(f"/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/models2/model_{i}.h5")
#     bleu = evaluate_model(model, val_descriptions, val_features, tokenizer, max_length)
#     val_bleu_scores.append(bleu)
#     print(f"Epoch {i}: BLEU-1={bleu[0]:.4f}, BLEU-2={bleu[1]:.4f}, BLEU-3={bleu[2]:.4f}, BLEU-4={bleu[3]:.4f}")


# bleu_4_scores = [score[3] for score in val_bleu_scores]

# plt.plot(range(1, 11), bleu_4_scores, marker='o')
# plt.title("Validation BLEU-4 Score Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("BLEU-4 Score")
# plt.xticks(range(1, 11))
# plt.grid(True)
# plt.show()

# best_epoch = np.argmax(bleu_4_scores)

# Load test data
test_imgs = load_photos("/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/Flickr8k_text/Flickr_8k.testImages.txt")
test_descriptions = load_clean_descriptions("descriptions.txt", test_imgs)
test_features = load_features(test_imgs)

# Evaluate on test set (once only)
model = define_model(vocab_size, max_length)
# # --- Main ---

# # Load weights (not full model)
model.load_weights("/home/mukta-hacker/Desktop/Mukta/Projects/Image_captioning/kaggle_coco_118K/model_0.h5")
# # Load test data
test_img_list = load_photos(os.path.join(dataset_text, "Flickr_8k.testImages.txt"))
# test_img_list = load_photos(os.path.join(dataset_text, "Flickr_8k.devImages.txt"))
test_descriptions = load_clean_descriptions("descriptions.txt", test_img_list)
test_features = load_features(test_img_list)

# # Run evaluation
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
