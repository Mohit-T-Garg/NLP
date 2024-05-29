import argparse
import json
from sklearn.feature_extraction.text import  TfidfTransformer
from evaluation import compute_macro_f1_score, compute_micro_f1_score
from scipy.sparse import hstack
import pickle
from train import char_ngram_tokenizer

def run_inference(model_path, data_path):

    # Load test data
    with open(data_path, 'r', encoding='utf-8') as fp:
        loaded_data = json.load(fp)

    # Extract features and labels for validation set
    test_data = [sample['text'] for sample in loaded_data]

    # Load the saved model
    with open(f"{model_path}model.pkl", 'rb') as model_file:
        clf = pickle.load(model_file)

    # Load the vectorizer for words
    with open(f"{model_path}vectorizer.pkl", 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Load the vectorizer for subwords
    with open(f"{model_path}vectorizer_subwords.pkl", 'rb') as vectorizer_subwords_file:
        vectorizer_subwords = pickle.load(vectorizer_subwords_file)


    # Load the fitted TfidfTransformer
    with open(f"{model_path}tfidf_transformer.pkl", 'rb') as tfidf_transformer_file:
        tfidf_transformer = pickle.load(tfidf_transformer_file)

    # Feature extraction and transformation for validation set
    test_features_1 = vectorizer.transform(test_data)
    test_features_2 = vectorizer_subwords.transform(test_data)
    test_features = hstack([test_features_1, test_features_2])

    test_features = tfidf_transformer.transform(test_features)

    # Make predictions on the validation set
    predicted_test_labels = clf.predict(test_features)

    # Save the predicted labels to output.txt
    with open("output.txt", "w") as output_file:
        for label in predicted_test_labels:
            output_file.write(f"{label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language classification model.")
    parser.add_argument("--model", required=True, help="Path to the saved model componens.")
    parser.add_argument("--test_data", required=True, help="Path to the test JSON file.")
    args = parser.parse_args()

    run_inference(args.model, args.test_data)