"""Main program implementing the MLP class"""
import argparse
from models.MLP import MLP
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

BATCH_SIZE = 128
LEARNING_RATE = 1e-2
NUM_CLASSES = 2
NUM_NODES = [500, 500, 500]

def parse_args():
    parser = argparse.ArgumentParser(
        description="MLP written using TensorFlow 2.x, for Wisconsin Breast Cancer Diagnostic Dataset"
    )
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-n", "--num_epochs", required=True, type=int, help="number of epochs"
    )
    group.add_argument(
        "-r",
        "--result_path",
        required=True,
        type=str,
        help="path where to save actual and predicted labels array",
    )
    arguments = parser.parse_args()
    return arguments

def main(arguments):
    # load the features of the dataset
    features = datasets.load_breast_cancer().data

    # standardize the features
    features = StandardScaler().fit_transform(features)

    # get the number of features
    num_features = features.shape[1]

    # load the labels for the features
    labels = datasets.load_breast_cancer().target

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.30, stratify=labels
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(BATCH_SIZE)

    model = MLP(
        alpha=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        node_size=NUM_NODES,
        num_classes=NUM_CLASSES,
        num_features=num_features,
    )

    model.train(
        num_epochs=arguments.num_epochs,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    # Example of saving predictions
    # predictions = model.predict(test_dataset)  # Assuming you have a predict method in MLP class
    # model.save_labels(predictions=predictions, actual=test_labels, result_path=arguments.result_path,
