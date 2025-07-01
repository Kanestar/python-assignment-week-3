# Week 3: Machine Learning, Deep Learning, and NLP Tasks

This folder contains solutions to various machine learning, deep learning, and natural language processing tasks, as well as a deployment example.

## Table of Contents
1.  [Task 1: Classical ML with Scikit-learn (Iris Species Classification)](#task-1-classical-ml-with-scikit-learn-iris-species-classification)
2.  [Task 2: Deep Learning with TensorFlow/Keras (MNIST Handwritten Digits Classification)](#task-2-deep-learning-with-tensorflowkeras-mnist-handwritten-digits-classification)
3.  [Task 3: NLP with spaCy (Amazon Product Reviews - NER & Sentiment)](#task-3-nlp-with-spacy-amazon-product-reviews---ner--sentiment)
4.  [Deployment: MNIST Classifier with Streamlit](#deployment-mnist-classifier-with-streamlit)

## General Setup

Before running any of the scripts, ensure you have Python installed (Python 3.7+ is recommended). If not, download it from [python.org](https://python.org) and make sure to check "Add Python to PATH" during installation.

All necessary Python packages are listed in `requirements.txt`. To install them, navigate to the `week 3` folder in your terminal and run:

```bash
pip install -r requirements.txt
```

---

## 1. Task 1: Classical ML with Scikit-learn (Iris Species Classification)

**Goal:** Preprocess data, train a decision tree classifier to predict iris species, and evaluate using accuracy, precision, and recall.

**Dataset:** Iris Species Dataset (loaded directly from Scikit-learn).

**File:** `iris_classification.py`

**How to Run:**
1.  Navigate to the `week 3` folder in your terminal:
    ```bash
    cd week 3
    ```
2.  Run the Python script:
    ```bash
    python iris_classification.py
    ```

**Expected Output:**
 The script will print messages about data loading, preprocessing, model training, and then display the accuracy, weighted precision, weighted recall, and a detailed classification report for each Iris species.

---

## 2. Task 2: Deep Learning with TensorFlow/Keras (MNIST Handwritten Digits Classification)

**Goal:** Build a CNN model to classify handwritten digits, aim for >95% test accuracy, and visualize the model's predictions on 5 sample images.

**Dataset:** MNIST Handwritten Digits (loaded directly from Keras).

**File:** `mnist_cnn.py`

**How to Run:**
1.  Navigate to the `week 3` folder in your terminal:
    ```bash
    cd week 3
    ```
2.  Run the Python script:
    ```bash
    python mnist_cnn.py
    ```

**Expected Output:**
 The script will show progress during model training (epochs, loss, accuracy). After training, it will print the final test loss and accuracy. A Matplotlib window will then pop up displaying 5 random test images with their true and predicted labels. Close this window to finish the script execution.

**Note:** This script also saves the trained model as `mnist_cnn_model.h5` in the `week 3` folder if the accuracy goal is met. This model is used by the Streamlit deployment.

---

## 3. Task 3: NLP with spaCy (Amazon Product Reviews - NER & Sentiment)

**Goal:** Perform Named Entity Recognition (NER) to extract product names and brands, and analyze sentiment (positive/negative) using a rule-based approach.

**Text Data:** Sample user reviews from Amazon Product Reviews (defined within the script).

**File:** `nlp_spacy.py`

**How to Run:**
1.  Navigate to the `week 3` folder in your terminal:
    ```bash
    cd week 3
    ```
2.  **Download spaCy Language Model (one-time setup):**
    ```bash
    python -m spacy download en_core_web_sm
    ```
3.  Run the Python script:
    ```bash
    python nlp_spacy.py
    ```

**Expected Output:**
 The script will print the results of NER, showing extracted entities, potential brands, and product names for each review. Following that, it will display the sentiment (Positive/Negative/Neutral) for each review based on the rule-based analysis.

---

## 4. Deployment: MNIST Classifier with Streamlit

**Goal:** Deploy your trained MNIST classifier as a web interface using Streamlit.

**Prerequisites:**
*   You must have successfully run `mnist_cnn.py` (Task 2) at least once, and it should have saved the `mnist_cnn_model.h5` file in the `week 3` folder.

**Files:**
*   `mnist_app.py` (the Streamlit application)
*   `mnist_cnn_model.h5` (the pre-trained model)

### Running Locally (for testing)
1.  Navigate to the `week 3` folder in your terminal:
    ```bash
    cd week 3
    ```
2.  Run the Streamlit application:
    ```bash
    streamlit run mnist_app.py
    ```
    Your default web browser will open to `http://localhost:8501/` (or a similar address) displaying the app.

### Deploying to Streamlit Community Cloud (for sharing)
This method allows you to deploy your app directly from a GitHub repository for free.

1.  **Create a GitHub Repository:**
    *   Create a new **public** GitHub repository (e.g., `ml-dl-nlp-tasks`).
    *   **Add your `week 3` files to this repository.** The essential files for deployment are:
        *   `week 3/mnist_app.py`
        *   `week 3/mnist_cnn_model.h5`
        *   `week 3/requirements.txt`
    *   Push these files to your GitHub repository.

2.  **Sign up for Streamlit Community Cloud:**
    *   Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

3.  **Deploy Your App:**
    *   Click on **"New app"**.
    *   Provide the following details:
        *   **Repository:** Select your GitHub repository.
        *   **Branch:** Select the branch (e.g., `main`).
        *   **Main file path:** `week 3/mnist_app.py` (This is crucial! It tells Streamlit where your app script is within your repo).
        *   **Python version:** Choose a compatible Python version.
    *   Click **"Deploy!"**
    *   Streamlit will build and deploy your application. Once complete, you'll receive a unique URL to access and share your live app.

---

## File Structure of `week 3`

```
week 3/
├── iris_classification.py
├── mnist_cnn.py
├── mnist_app.py
├── nlp_spacy.py
├── requirements.txt
└── README.md
``` 