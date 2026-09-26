# Machine Learning Algorithms from Scratch

## Implementing Core Machine Learning Algorithms with Python and NumPy

This project implements several fundamental machine learning algorithms from scratch using Python and NumPy, including Decision Trees, k-Nearest Neighbors, Naive Bayes, Random Forests, and Neural Networks. The implementations demonstrate how these algorithms work internally and how they can be applied to classification problems and evaluated on real datasets.

## Installation

Clone the repository:

git clone https://github.com/ckraus27/machine-learning-algorithms-from-scratch.git
cd machine-learning-algorithms-from-scratch

Install the required Python libraries:

pip install numpy pandas matplotlib scikit-learn

## Usage

Each directory contains the implementation and supporting files for a different machine learning algorithm.

- decision_tree_and_knn/ — Decision Tree and k-Nearest Neighbors
- naive_bayes/ — Naive Bayes
- random_forest/ — Random Forest
- neural_network/ — Neural Network

Run the Python source files within each directory to train and evaluate the corresponding algorithms on the included datasets.

## Algorithms

- Decision Tree — Builds decision trees using entropy and information gain, supporting both categorical and numerical features.
- k-Nearest Neighbors — Classifies data points based on the nearest training examples.
- Naive Bayes — Implements probabilistic classification using the Naive Bayes approach.
- Random Forest — Builds an ensemble of decision trees using bootstrap sampling and random feature selection, then combines predictions through majority voting.
- Neural Network — Implements a neural network with forward propagation, backpropagation, and gradient-based weight updates.

## Model Evaluation

The implementations use several evaluation techniques to measure model performance.

- Stratified k-fold cross-validation
- Accuracy
- Precision
- Recall
- F1-score
- Hyperparameter evaluation
- Performance visualizations

The Random Forest implementation evaluates different numbers of trees and compares model performance across the WDBC and loan datasets.

## Development

To work with the project locally, clone the repository and install the required dependencies as described above. The implementations are organized by algorithm so that each component can be examined and modified independently.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Contributor Expectations

This project was developed as a personal academic project. Contributions are not currently being solicited.

## Known Issues

- Some implementations may require the included datasets to be located in their expected directories.
- Individual algorithms may have limitations depending on the datasets and configurations used for evaluation.