# Sentiment Analyzer — Basic Machine Learning Edition

A beginner-friendly three-class sentiment analysis project. It uses a trained
Scikit-learn Logistic Regression classifier, TF-IDF text features, and a Flask
web application.

## Sentiment classes
- Positive
- Negative
- Neutral

## Project files
- `dataset.csv` — small labelled example dataset for learning and demonstration
- `train_model.py` — splits data, trains the model, prints evaluation metrics, and saves it
- `sentiment_model.pkl` — generated trained model (created after training)
- `app.py` — Flask backend and prediction API
- `templates/index.html` — webpage
- `static/style.css` — styling
- `static/script.js` — browser interaction

## Run locally

Use Python 3.10 or newer. Open a terminal in this project folder.

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

This prints accuracy, precision, recall, F1-score, and a confusion matrix.
It also creates `sentiment_model.pkl`.

Start the web application:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Important academic note

The included CSV is a small illustrative teaching dataset, not a large
benchmark dataset. Metrics from a small dataset are only a demonstration and
should not be presented as real-world performance. For a stronger assignment,
replace or expand `dataset.csv` with a larger, properly labelled dataset.
Keep the columns named `text` and `sentiment`, and use only the labels
`positive`, `negative`, and `neutral`.

The confidence values are the model's predicted class probabilities; they
are not a guarantee that the prediction is correct.

## Methodology

1. Load labelled text examples from CSV.
2. Split the data into training and testing sets using stratification.
3. Convert text into TF-IDF numerical features (unigrams and bigrams).
4. Train a Logistic Regression classifier.
5. Evaluate predictions on held-out test data.
6. Save the fitted pipeline and use it through Flask for new text.

## Limitations

The model may struggle with sarcasm, slang, mixed-language text, context,
negation, and domain-specific expressions. Model quality depends strongly
on the size, balance, and quality of the labelled dataset.

## Training results (this run)

- Dataset: `dataset.csv` = 60 original hand-written rows (`dataset_original.csv`) + ~234 template-generated
  rows (`expand_dataset.py`). The generated rows are synthetic and repetitive, so the scores below are
  optimistic and only demonstrate the pipeline.
- 5-fold CV accuracy: ~92.5%; held-out test accuracy: ~93.2% (59 samples). Full output: `training_output.txt`.
- On the original 60 rows alone the same approach scored only ~33-40%, which shows how data-limited it was.
