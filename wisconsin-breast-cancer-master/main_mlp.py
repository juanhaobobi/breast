"""Main program implementing the MLP class"""
import argparse
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.data import Dataset
from models.MLP import MLP

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

def load_data():
    features = datasets.load_breast_cancer().data
    features = StandardScaler().fit_transform(features)
    labels = datasets.load_breast_cancer().target
    return features, labels

def main(arguments):
    features, labels = load_data()
    num_features = features.shape[1]

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.30, stratify=labels
    )

    train_dataset = Dataset.from_tensor_slices((train_features, train_labels)).shuffle(len(train_features)).batch(BATCH_SIZE)
    test_dataset = Dataset.from_tensor_slices((test_features, test_labels)).batch(BATCH_SIZE)

    model = MLP(
        alpha=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        node_size=NUM_NODES,
        num_classes=NUM_CLASSES,
        num_features=num_features,
    )

    model.train(num_epochs=arguments.num_epochs, train_dataset=train_dataset, test_dataset=test_dataset)

    # 保存预测结果
    predictions = model.predict(test_dataset)  # Assuming you have a predict method in MLP class
    model.save_labels(predictions=predictions, actual=test_labels, result_path=arguments.result_path, phase='testing', step=arguments.num_epochs)

if __name__ == "__main__":
    args = parse_args()
    main(args)

