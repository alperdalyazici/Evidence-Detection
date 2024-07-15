# Bidirectional LSTM and RoBERTa for Evidence Detection
## Overview
This project implements a Bidirectional Long Short-Term Memory (BiLSTM) and RoBERTa model for evidence detection, which aims to determine the relevance of evidence to a given claim. 

## Code Structure
### Code Structure for Bidirectional LSTM Model
The code is organized into the following main components:

Data Preparation: Loading and preprocessing of training and testing data.
Model Definition: Construction of the BiLSTM model architecture using TensorFlow/Keras.
Hyperparameter Tuning: Optimization of model hyperparameters using the Keras Tuner library.
Model Training: Training the BiLSTM model on the training data.
Model Evaluation: Evaluation of the trained model on the development/validation dataset.
Prediction: Generating predictions on the test dataset and saving them to a CSV file.

### Code Structure for RoBERTa Model:

The code is organized into the following main components:

Data Preparation: Loading and preprocessing of training and testing data.
Model Definition: Implementing the pre-trained RoBERTa model using the Hugging Face Transformers library.
Hyperparameter Tuning: Optimization of model hyperparameters by iteratively training and evaluating the model.
Model Training: Fine-Tuning the RoBERTa model on the training data.
Model Evaluation: Evaluation of the trained model on the development/validation dataset.
Prediction: Generating predictions on the test dataset and saving them to a CSV file.

## Running Instructions
### Running Instructions for Bidirectional LSTM Model
To run the training and evaluating code, follow these steps:
Open the training and evaluation notebook file (train.ipynb).
If the data files need to be uploaded, use the Colab interface to upload them manually, such as train.csv and dev.csv.
Run the code cells in the notebook sequentially, which will execute the entire pipeline, including data loading, preprocessing, model training and evaluation.

To run the prediction code, follow these steps:
Open the prediction notebook file (demo.ipynb).
If the trained model is not stored in Google Drive and needs to be uploaded, you can use the Colab interface to upload it manually. 
If the data files need to be uploaded, use the Colab interface to upload them manually, such as train.csv and test.csv.
Run the code cells in the notebook sequentially, which will execute the entire pipeline for making predictions on the test data.

### Running Instructions for RoBERTa Model:

To run the training and evaluating code, follow these steps:
Open the training and evaluation notebook file (roberta_training_evaluation.ipynb).
If the data files need to be uploaded, use the Colab interface to upload them manually, such as train.csv and dev.csv.
Run the code cells in the notebook sequentially, which will execute the entire pipeline, including data loading, preprocessing, model training and evaluation.

To run the prediction code, follow these steps:
Open the prediction notebook file (roberta_democode.ipynb).
If the trained model is not stored in Google Drive and needs to be uploaded, you can use the Colab interface to upload it manually. 
If the data files need to be uploaded, use the Colab interface to upload them manually, such as train.csv and test.csv.
Run the code cells in the notebook sequentially, which will execute the entire pipeline for making predictions on the test data.

## Attribution:

### Code Attribution:

The hyperparameter tuning process was implemented using the Keras Tuner library (https://keras-team.github.io/keras-tuner/).
The RandomSearch tuner was adapted from the Keras Tuner documentation and examples.
The RoBERTa model was implemented using the Hugging Face Transformers library (https://huggingface.co/transformers/).
The RoBERTa fine-tuning process was adapted from the Hugging Face Transformers documentation (https://huggingface.co/docs/transformers/model_doc/roberta)


### Data Attribution:

The training data (train.csv) and development data (dev.csv) were obtained from https://online.manchester.ac.uk/webapps/blackboard/content/listContent.jsp?course_id=_78254_1&content_id=_15034148_1&mode=reset with permission.
The testing data (test.csv) was obtained from (https://online.manchester.ac.uk/webapps/blackboard/content/listContent.jsp?course_id=_78254_1&content_id=_15034148_1&mode=reset) with permission.

### Libraries Attribution:

The code utilizes the following libraries:

Pandas (https://pandas.pydata.org/)
NumPy (https://numpy.org/)
TensorFlow (https://www.tensorflow.org/)
Keras Tuner (https://keras-team.github.io/keras-tuner/)
Scikit-learn (https://scikit-learn.org/)
Hugging Face Transformers (https://huggingface.co/transformers/)
RoBERta (https://huggingface.co/docs/transformers/model_doc/roberta)

## Model Storage
Model Storage for Bidirectional LSTM Model
Trained models can be accessed and downloaded from the following links:
https://drive.google.com/file/d/1--nnPcu083PI_RFxxeZaaK9YmQ1dtQ9m/view?usp=sharing

Model Storage for RoBERTA Model
Trained models can be accessed and downloaded from the following links:
https://drive.google.com/drive/folders/1VrTgFC6gOMRr_SP5JwtmBUAhT115liov?usp=sharing
