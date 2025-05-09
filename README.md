# Multimodal Sarcasm Detection using Hierarchical Fusion

This project implements a Hierarchical Fusion Model to detect sarcasm in tweets using **text**, **image**, and **attribute** modalities.

##  Overview

Sarcasm is a nuanced form of expression that often requires visual and contextual understanding. This model leverages:
- BiLSTM for text encoding
- ResNet for image feature extraction
- Attribute detection for object and color cues
- A multi-level fusion mechanism to combine all three modalities for final classification

---

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/anoop7384/sarcasm-detection.git
cd sarcasm-detection
````

### 2. Install Dependencies

Make sure you are using **Python 3.7+** and have `pip` installed.

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Download the **Twitter Multimodal Sarcasm Dataset** from the following link:

 [Dataset Link](https://github.com/headacheboy/data-of-multimodal-sarcasm-detection)

After downloading, place the dataset folder in the project root:

```
multimodal-sarcasm/
├── data/
│   ├── train.json
│   ├── test.json
│   └── val.json
```

---

##  Running the Model

Once setup is complete, simply run the following:

```bash
python main.py
```

This will:

* Load the dataset
* Preprocess images and text
* Train and evaluate the hierarchical fusion model
* Print out performance metrics

---


