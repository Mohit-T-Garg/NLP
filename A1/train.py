import json
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import ComplementNB
from scipy.sparse import hstack
import pickle

# Define a custom tokenizer for character n-grams (sub-word-level features)
def char_ngram_tokenizer(text):
    ngrams_2 = [text[i:i+2] for i in range(len(text) - 1)]  # Use 2-character n-grams 
    ngrams_1 = [text[i:i+5] for i in range(len(text) - 4)]  # Use 5-character n-grams 
    ngrams_ = [text[i:i+4] for i in range(len(text) - 3)]  # Use 4-character n-grams 
    ngrams = [text[i:i+3] for i in range(len(text) - 2)]  # Use 3-character n-grams 
    return ngrams + ngrams_ + ngrams_1 + ngrams_2

# Train model using training data given and save model and other related files to given location.
def train_model(data_path, save_path):

    # Load training data
    with open(f"{data_path}train.json", 'r', encoding='utf-8') as fp:
        train_data = json.load(fp)

    # Extract features and labels for training set
    X_train = [sample['text'] for sample in train_data]
    y_train = [sample['langid'] for sample in train_data]

    # Load validation data
    with open(f"{data_path}valid.json", 'r', encoding='utf-8') as fp:
        valid_data = json.load(fp)

    # Extract features and labels for validation set
    X_valid = [sample['text'] for sample in valid_data]
    y_valid = [sample['langid'] for sample in valid_data]

    # Load validation data
    with open(f"{data_path}valid_new.json", 'r', encoding='utf-8') as fp:
        valid_data_new = json.load(fp)

    # Extract features and labels for validation set
    X_valid_new = [sample['text'] for sample in valid_data_new]
    y_valid_new = [sample['langid'] for sample in valid_data_new]

    # Generate new training data
    train_data = X_train # + X_valid_new + X_valid
    train_labels = y_train # + y_valid_new + y_valid

    # Basic info about classes distribution
    classes_distribution = {label: train_labels.count(label) for label in set(train_labels)}
    print("Classes Distribution:")
    print(classes_distribution)

    # Calculate average size of features
    average_feature_size = sum(len(sample.split()) for sample in train_data) / len(train_data)
    print("\nAverage Size of Features:")
    print(average_feature_size)

    # Feature extraction: Convert text data to a matrix of token counts, include bigrams, remove stop words, and add regex features
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    train_features_1 = vectorizer.fit_transform(train_data)

    # Feature extraction: Sub-word level features are generated using a custom character n-gram tokenizer (n = 3)
    vectorizer_subwords = CountVectorizer(analyzer='word', tokenizer=char_ngram_tokenizer)
    train_features_2 = vectorizer_subwords.fit_transform(train_data)

    # Get all Features
    train_features = hstack([train_features_1, train_features_2])

    # Feature transformation: Convert token counts to TF-IDF representation
    tfidf_transformer = TfidfTransformer()
    train_features = tfidf_transformer.fit_transform(train_features)

    # Report no. of features
    print("\nNumber of Features:")
    print(train_features.shape[1])

    # Train a Compliment Naive Bayes classifier
    clf = ComplementNB(alpha = 0.025, class_prior=None, fit_prior=False)
    clf.fit(train_features, train_labels)

    # Save the trained model and related files
    with open(f"{save_path}model.pkl", 'wb') as model_file:
        pickle.dump(clf, model_file)

    with open(f"{save_path}vectorizer.pkl", 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    with open(f"{save_path}vectorizer_subwords.pkl", 'wb') as vectorizer_subwords_file:
        pickle.dump(vectorizer_subwords, vectorizer_subwords_file)

    with open(f"{save_path}tfidf_transformer.pkl", 'wb') as tfidf_transformer_file:
        pickle.dump(tfidf_transformer, tfidf_transformer_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language classification model.")
    parser.add_argument("--data", required=True, help="Path to the data JSON file.")
    parser.add_argument("--save", required=True, help="Path to the directory to save the trained model and related files.")
    args = parser.parse_args()

    train_model(args.data, args.save)

