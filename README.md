# COVID-19_Symptoms_Detection
In this project, I used a Covid Dataset from Kaggle and applied different types of Machine Learning algorithms to classify whether a patient is Covid positive or negative based on the existing symptoms.

Here is the source of our Dataset: https://www.kaggle.com/zhiruo19/covid19-symptoms-classification

1999 Patients information is available here with 6 attributes. The attributes are: - fever, bodypain, age, runnynose, diffbreath, infected

*Step 1: Data Preparation:* We will use Covid-19 symptoms classification dataset from Kaggle created by Zhi Ruo which is publicly available. Whether a patient is covid-19 positive or negative is the goal of this algorithm. To accomplish this we are going to use a machine learning classification algorithm so that it can predict the ultimate class of unknown input.

We will also check if there are any missing or null data points in the dataset. Different types of encoding methods are used to convert categorical data. As all the attributes of this dataset are numeric, we do not have to perform this task. We will use the matplotlib library to visualize the different features of the data. Most of the dataset will contain highly varying features in magnitudes, units, and range. As machine learning methods use Euclidean distance in their calculations, we have to use the scaling method to bring all features to the same level of magnitudes. That is why, we will transform our data so that it can fit within a range, like 0 – 1 or 0 – 100.

Step 2: Splitting the Dataset: After preparing the dataset we have to split the dataset into training and testing set. The training set will contain a known output and using this training set model will learn. After that, we will use the testing data set to evaluate our model’s prediction on the sample. For dividing the dataset into 2 sub-groups (training and testing) we can use the scikit-learn library or it can be any simple procedure.

Step 3: Model Selection: There are different kinds of Machine Learning algorithms that are usually used by data scientists for large data sets. These algorithms can be classified into two sets: supervised learning and unsupervised learning. Regression and Classification problems are the two subgroups of supervised learning problems. A classification problem can be defined when the output is a category like detecting cancer cells “malignant” or “benign”. In this Covid-19 dataset, we have the Dependent variable i.e a patient having only two different sets of values, either 1 (Covid-19 Positive) or 0 (Covid-19 Negative). So we will use different Classification algorithms to classify whether a patient is infected or not. We have different types of classification algorithms in Machine Learning:-

1. Logistic Regression
2. Nearest Neighbor
3. Support Vector Machines
4. Kernel SVM
5. Naïve Bayes
6. Decision Tree Algorithm
7. Random Forest Classification

We will analyze our dataset and apply different classification algorithms. By using the scikit-learn library we will import all the methods of classification algorithms and check which model is better than others.

Step 4: Model Evaluation: Now our model will predict the result of the testing set and we have to check the accuracy of each model. To visualize the performance of an algorithm, the confusion matrix is widely used. We will evaluate the model to get the classification result from the confusion matrix. Classification accuracy will be used to find the actual performance of each class of the model. Basically, classification accuracy is the ratio of the number of correctly classified samples and the total number of samples.

Step 5: Improving the Model: After applying the different classification models, we will choose the best classification model for this dataset and do further calculations to improve our model.

At first, we calcualte the predicted result using cross validation and then we make ensemble of classifiers and then average the result to check whether the accuracy is increased or not.
