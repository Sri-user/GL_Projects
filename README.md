# GL_Projects
Projects done as part of my PGP on Machine Learning and Artificial Intelligence at GreatLakes

Zip file contains the dataset used for each project

Please click Add files to upload link in each .pynb file to view the project description of each model

# Project 1 - Statistics :
Applied statistics techniques learned to leverage more information on the insurance dataset, which can improve the existing strategy of the company to target people for medical insurance

   Data: Statistics.zip
   Ipython code: Sritharan_Statistics Assignment.ipynb

# Project 2 - Classification Model :
Objective is to identify pattern in the data and decide who is more likely to get a personal loan

   Data: SupervisedLearning_ClassificationModel.zip.
   
  The file Bank.xls contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the           customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan         campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the       earlier campaign.

  Ipython code: Project - PersonalLoanModelling_ClassificationModel.ipynb
  
  Analysed and visualized data and cleansed the data to remove the columns which were not needed for the modelling. Used Logistic         regression and naive bayes to model the binary classification model to predict the potential loan customers

# Project 3 - Unsupervised Learning :
The purpose is to classify a given silhouette as one of three types of vehicle, using a set of features extracted from the silhouette. The vehicle may be viewed from one of many different angles

  Data: UnsupervisedLearning_PCA_SVM.zip
  
  Classify a given silhouette as one of four different types of vehicle, using a set of features extracted from the silhouette. The       vehicle may be viewed from one of many different angles.

  Four "Corgie" model vehicles were used for the experiment: a double decker bus, Cheverolet van, Saab 9000 and an Opel Manta 400 cars.   This particular combination of vehicles was chosen with the expectation that the bus, van and either one of the cars would be readily   distinguishable, but it would be more difficult to distinguish between the cars.

  Ipython code: PCA_Techiniques_ClassificationModel.ipynb
  
  Actual data does not contain names for the feature. Columns names have been provided here for academic purposes. Cleansed, scaled,       replaced/modified missing values and removed columns based on the analysis of the data. Further PCA is used to know the varirance       explained by each feature and kept only the features that explain more than 95 percent of the variance. Built a multiclassification     model using SVM and KNN
  
# Project 4 - Supervised Learning - Regression
Modeling of strength of high performance concrete using Machine Learning

   Data - FeatureEngineeringTechniques_Regression.zip
   The concrete compressive strength is a highly nonlinear function of age and ingredients. These ingredients include cement, blast        furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.
   
   Ipython code - FeautureEngineering.ipynb
   Handled outliers and missing values and build regression models using Linear regression. Created new clusters within the data by        analysing the data using K-means and built model for each each cluster using random forest regressor. Used cross validation              techniques and gridsearchCV to find and tune the best hyperparameters for the model
 
# Project 5 - Supervised Learning - Classification
Based on the previous marketing campaigns, predict whether a customer is more likely to do a term deposit in bank

   Data - EnsemblingTechiniques.zip
   The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone    calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be      ('yes') or not ('no') subscribed.
   
   Ipython code - EnsembledTechniques_Project.ipynb
   Used Gradient and Ada boosting on the existing decision tree model built, to improve the precision and recall for the classes
   
# Project 6 - Neural Networks - Digit recognition
SVHN is a real-world image dataset for developing machine learning and objectrecognition algorithms with minimal requirement on data formatting but comes from asignificantly harder, unsolved, real world problem (recognizing digits and numbers innatural scene images). SVHN is obtained from house numbers in Google Street Viewimages

   Data - https://drive.google.com/file/d/1L2-WXzguhUsCArrFUc8EEkXcj33pahoS/view
   
   Ipython code - DL_SVNH.ipynb
   Created a sequential model with keras using RELU layers and softmax layer at the output to classicy 10 numbers (0-9). Modified the      image using numpy to remove the unwanted pieces of image to improve the accuracy. Implemented batch normalization for training the      neural network. Created a confusiion matrix for each class and have displayed precision, recall and f1 score for each class

# Project 7 - Face Detection
Goal is to build a face detection model which includes building a face detector to locate the position of a face in an image
   
   Data - https://drive.google.com/file/d/1rolp8QqyKkvxJwlBAPPGtG2f3JA7hMwk/view
   
   Ipython code - FaceDetection_GL.ipynb
   Analyzed and visualized the images which were in .npy format. Created an numpy array with1’s in the faces and 0’s elsewhere for each    image, using the metadata bounding box information present in the .npy dataset (Ytrain data). Used mobile net model for object          detection from keras, and used transfer learning to make the last 6 layers trainable and freezed the remaining layers. Implemented a    UNet architecture and up sampled the layers from mobilenet

# Project 8 - Face Recognition
Recoginise and find difference between the faces of hollywood stars using CNN 
   
   Data - https://drive.google.com/file/d/1AuJ7yQlq3FhRFZy3MK2DwvtxtFgTDP2h/view
   This dataset contains 10.770 images for 100 people. All images are taken from 'Pinterest' and aligned using dlib library
   
   Ipython code - FaceRecognition_GL.ipynb
   We use a pre-trained model trained on Face recognition to recognize similar faces and find whether two given faces are of the same      person or not. Created an embedding for each image, mapping face features as a vector representation. Later an euclidean distance        method is employed to calculate the distance and predict how similar a test image is to the trained image. Created a training and        test data from the images and scaled these images using Standard Scaler from scikit learn. Features from embedding were analysed        using PCA to find the cumulative variance and reduced the number of features which would explain more than 95 percent of the variance    in each image. Implemented SVM classifier on the already trained VGG Face to predict/recognize the test image
   Achieved a 96 percent accuracy of recognizing faces
   
# Project 9 - Binary classification of sentiment based on IMBD review comments using sequential NLP

   Data - keras imdb dataset
   
   Ipython code -  IMBD_MovieReview_NLP.ipynb
   Got the word index and then created a key-valuepair for word and word_id.Build a Sequential Model using Keras for the                    Sentiment Classification task. GeneratedWord Embeddings and retrieved outputs of each layer with Keras
   
# Project 10 - Sarcasm detection using Birectional LSTM

   Data - https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection
   
   Ipython code - SarcasmDetection.ipynb
   Tokenized the words from each sentence using Tokenizer class. Created an word embedding using the existing Glove embedding of            Wikipedia. Created an embedding matrix with values matching our glove embedding with the word to index obtained from the tokenizer      class Implemented a Bidirectional LSTM model to predict the sarcasm in the sentence
   









