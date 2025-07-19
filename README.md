# 📸 Image Captioning using CNN + LSTM on Flickr8K Dataset

This project demonstrates an end-to-end pipeline for **image caption generation**, combining **pretrained CNN (Xception)** for visual feature extraction and **LSTM** for natural language generation. The model is trained and evaluated on the **Flickr8K** dataset.

---

## 📂 Dataset

The project uses the [Flickr8K Dataset](https://drive.google.com/file/d/1u3oqx36XApnAykFDB6EEWUIfd_CxRQQ9/view) which contains ~8,000 images, each annotated with 5 captions.

You’ll also need the [Flickr8K Text Metadata](https://drive.google.com/file/d/1qcRy3WpQv4dGtu65gETtYLWxDPBrRtx1/view), which includes:
- `Flickr8k.lemma.token.txt`: Captions for each image.
- `Flickr_8k.trainImages.txt`: List of 6,000 training images.
- `Flickr_8k.devImages.txt`: 1,000 validation images.
- `Flickr_8k.testImages.txt`: 1,000 test images.

---

## 🧠 Workflow Overview

All core functionality is defined in `main.py` and `evaluate.py`. The steps include:

### 🔹 In `main.py`:
- **Data Preprocessing**: Load and clean captions, build vocabulary, and save processed data.
- **Feature Extraction**: Use pretrained Xception to extract 2048-D image features for all images.
- **Prepare Training Data**: Select only training image data and features.
- **Tokenizer Creation**: Map captions to sequences using Keras Tokenizer and save for reuse.
- **Sequence Generation**: Convert partial captions and features to input-output training pairs.
- **Model Definition & Training**: Define a dual-branch CNN+LSTM model and train for 10 epochs.

### 🔹 In `evaluate.py`:
- Load tokenizer, trained model, and test image features.
- Generate captions for test images and compare with ground truth.
- Compute BLEU-1 to BLEU-4 scores for evaluation.

---

## 📈 Results

- **Vocabulary Size**: 7,577 words  
- **Max Caption Length**: 32 tokens  
- **Trainable Parameters**: 5,002,649  
- **Training Epochs**: 10 (Loss reduced from **5.03 → 2.92**)

### 🏆 BLEU-4 Scores:
| Dataset     | BLEU-4 |
|-------------|--------|
| Validation  | **0.0574** (Best at Epoch 8) |
| Test        | 0.0513 (Best Epoch)          |
| Test        | 0.0478 (Last Epoch)          |

---

## ✅ How to Use

1. Download the dataset and place images and text files in appropriate folders.
2. Install "requirements.txt"
3. Run `main.py` to preprocess, extract features, and train the model.
4. Run `evaluate.py` to generate captions and evaluate BLEU scores.

> ⚙️ All necessary functions, training logic, and evaluation steps are explained in the documentation provided in this repository.

