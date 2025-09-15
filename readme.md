# Spam Email Classification: Scikit - PyTorch

This repository provides two end-to-end implementations of a spam email classification system, built with Scikit-learn and PyTorch. It showcases how to preprocess data, train models, evaluate performance, and serve predictions through APIs for real-world usage.

### Scikit-learn Components

#### MultinomialNB:

    A Naive Bayes classifier designed for discrete features such as word counts. It assumes that word occurrences follow a multinomial distribution, making it well-suited for text classification tasks like spam detection.

#### CountVectorizer:

    Converts raw text into a matrix of token counts. Each document is represented by the frequency of words it contains, serving as the input features for classifiers.

### PyTorch Components

#### TfidfVectorizer:

    Transforms text into numerical features using Term Frequency–Inverse Document Frequency. Unlike simple counts, it reduces the weight of common words and emphasizes rare but informative words.

#### Recurrent Neural Network (RNN):

    A type of neural network designed to handle sequential data. It processes input word embeddings step by step, capturing temporal dependencies in text, making it effective for spam/ham classification.

#### CrossEntropyLoss:

    A loss function commonly used for classification tasks. It measures the difference between predicted class probabilities and the true class labels.

#### Adam Optimizer:

    An adaptive optimization algorithm that combines the benefits of AdaGrad and RMSProp. It adjusts learning rates dynamically, leading to faster and more efficient training.

## Getting Started

### Installation

1. Clone the repository.
   ```sh
   git clone https://github.com/bosukeme/spam-email-classification-scikit-pytorch.git
   cd spam-email-classification-scikit-pytorch
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Data

- The dataset is located in

```sh
data/spam.csv

```

## Usage

### PyTorch Implementation

- Jupyter notebook for training and exploration: `spam_detection_pytorch.ipynb`.
- FastApi set up is in main.py.
- Model architecture and utilities are in models.py.
- Service functions for prediction are in services.py.

#### Run FastAPI

```sh
cd pytorch
uvicorn main:app --reload
```

### Scikit-learn Implementation

- Jupyter notebook for training and exploration: `spam_detection_scikit.ipynb`.
- Flask API set up is in app.py.
- Service functions for prediction are in services.py.

#### Run Flask

```sh
cd scikit
python app.py
```

## API Usage

#### Example Request (PyTorch / FastAPI)

```sh
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You have won a free prize."}'
```

#### Example Request (Scikit-learn / Flask)

```sh
curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Are we still on for lunch tomorrow?"}'

```

## Model Artifacts

- PyTorch model: spam_model_rnn.pth
- Scikit-learn model: spam_model_scikit.pkl
- Vectorizers for both implementations are stored in their respective `models/` folders.

## Notebooks

Interactive Jupyter notebooks are provided for exploration, visualization, and experimentation:

- `spam_detection_pytorch.ipynb`

- `spam_detection_scikit.ipynb`

## License

This project is for educational purposes.
