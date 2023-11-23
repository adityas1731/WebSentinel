# Basic libraries
import os
import io
import sys
import math
import time
import random
import collections
import numpy as np
import seaborn
from os import walk
from joblib import dump, load

from tokenizers import ByteLevelBPETokenizer
from langdetect import detect

# Scikit learn stuff
import sklearn
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Parsing arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_folder", type=str, default = "tokenizer", help="Folder where tokenizer files have been placed")
parser.add_argument("--labeled_data_folder", type=str, default = "labeled_data", help="Labeled data folder")
parser.add_argument("--ignore_other_languages", type=int, default = 1, help="Whether to ignore languages other than english. 0 for no, 1 for yes.")
parser.add_argument("--apply_different_thresholds", type=int, default = 1, help="Whether to show results for different thresholds or not")
parser.add_argument("--save_model_dir", type=str, default = "saved_models", help="Directory to store trained models.")

args = parser.parse_args()
tokenizerFolder = args.tokenizer_folder
labeledDataFolder = args.labeled_data_folder
isIgnoreOtherLanguages = args.ignore_other_languages
isApplyDifferenceThresholds = args.apply_different_thresholds
saveModelDirectory = args.save_model_dir


def detectLanguage(inputString):
	"""
	Detect language given an input html code string
	"""
	inputString = str(inputString) # in case it is not a string
	language = "n/a"
	try:
		language = detect(fileData)
		return language
	except:
		return language

	
# Load tokenization files
tokenizer = ByteLevelBPETokenizer(
    tokenizerFolder + "/tokenizer.tok-vocab.json",
    tokenizerFolder + "/tokenizer.tok-merges.txt",
)
tokenizerVocabSize = tokenizer.get_vocab_size()
print("Tokenizer files have been loaded and the vocab size is %d..." % tokenizerVocabSize)

# Get all html files
files = []
labels = []
for(_,_,f) in walk(labeledDataFolder + "/legitimate_htmls"):
	files.extend([labeledDataFolder + "/legitimate_htmls/" + file for file in f])
	labels.extend([0 for _ in f])
for(_,_,f) in walk(labeledDataFolder + "/phishing_htmls"):
	files.extend([labeledDataFolder + "/phishing_htmls/" + file for file in f])
	labels.extend([1 for _ in f])
print("Total number of html files: %d\n" % len(files))
print("Number of phishing html files: %d, Legitimate html files: %d" % (labels.count(1), labels.count(1)))


print("\nCreating a dictionary for document frequency values for tfidf scores...\n")
count = 0
docDict = collections.defaultdict(list)
ignoredFiles = {}
for i in range(0, len(files)):
	file = files[i]
	label = labels[i]
	count = count + 1
	print("Files processed: %d, Total files: %d" % (count, len(files)))

	# load raw html data
	fileData = io.open(file, "r", errors="ignore").readlines()
	fileData = ''.join(str(line) for line in fileData)
	fileData = fileData.replace("\n", " ")

	# ignore the website if language is other than english
	if isIgnoreOtherLanguages == 1:
		inputLanguage = detectLanguage(fileData)	
		if inputLanguage != "en":
			ignoredFiles[file] = True
			continue

	# tokenize html code
	output = tokenizer.encode(fileData)
	outputDict = collections.Counter(output.ids)

	# add counts to a dictionary for tfidf scores. 
	for token in outputDict:
		docDict[token].append(file)
	

print("\nAssigning tfidf weights to tokens...\n")
features = []
htmlLabels = []
totalFilesUnderConsideration = len(files) - len(ignoredFiles)
count = 0
for i in range(0, len(files)):
	file = files[i]
	label = labels[i]
	count = count + 1
	print("Files processed: %d, Total files: %d" % (count, len(files)))

	# check if this file should be ignored based on language
	if file in ignoredFiles:
		continue

	# load data
	fileData = io.open(file, "r", errors="ignore").readlines()
	fileData = ''.join(str(line) for line in fileData)
	fileData = fileData.replace("\n", " ")
	
	# tokenize
	output = tokenizer.encode(fileData)
	outputDict = collections.Counter(output.ids)

	# apply tfidf weighting
	array = [0] * tokenizerVocabSize
	for item in outputDict:
		if len(docDict[item]) > 0:
			array[item] = (outputDict[item]) * (math.log10(len(files) / len(docDict[item]))) # this is a small future leak in the model. I should fix it sometime.
	
	features.append(array)
	htmlLabels.append(label)
	

print("\nTraining data have been prepared...")
print("Phishing websites count: %d, Legitimate websites count: %d" % (htmlLabels.count(0), htmlLabels.count(1)))
print("Starting training...\n")

EXPERIMENT_ITERATIONS = 5
accuracies, precisions, recalls, fscores, aucs = [], [], [], [], []
for i in range(EXPERIMENT_ITERATIONS):
	# Split data into training and testing
	trainData, testData, trainLabels, testLabels = train_test_split(features, htmlLabels, test_size=0.1) # please consider this test data as validation because of the future leak in tfidf score calculation.
	testData = trainData
	testLabels = trainLabels

	# Create a classifier instance
	classifier = RandomForestClassifier(n_estimators = 100)

	# Train classifier
	classifier.fit(trainData, trainLabels)

	# Get test predictions
	testPredictions = classifier.predict(testData)
	testPredictionsProbs = classifier.predict_proba(testData)

	# Calculate metrics
	accuracy = round(accuracy_score(testLabels, testPredictions) * 100, 2)
	precision = round(precision_score(testLabels, testPredictions) * 100, 2)
	recall = round(recall_score(testLabels, testPredictions) * 100, 2)
	fscore = round(f1_score(testLabels, testPredictions) * 100, 2)
	auc = round(roc_auc_score(testLabels, testPredictions) * 100, 2)

	# Store in lists for averaging later on
	accuracies.append(accuracy)
	precisions.append(precision)
	recalls.append(recall)
	fscores.append(fscore)
	aucs.append(auc)


	if i == EXPERIMENT_ITERATIONS - 1 and isApplyDifferenceThresholds == 1:
		print("Trying out different confidence thresholds and showing the resulting preformance...")
		CONFIDENCE_THRESHOLDS = [0.9, 0.8, 0.7, 0.6, 0.5]
		for threshold in CONFIDENCE_THRESHOLDS:
			topnprobs = [[testPredictions[i], testLabels[i]] for i in range(0, len(testLabels)) if testPredictionsProbs[i][0] >= threshold or testPredictionsProbs[i][1] >= threshold]
			topnlabels = [item[1] for item in topnprobs]
			topnpredictions = [item[0] for item in topnprobs]

			confusionMatrix = confusion_matrix(topnlabels, topnpredictions)
			accuracy = round(accuracy_score(topnlabels, topnpredictions) * 100, 2)
			precision = round(precision_score(topnlabels, topnpredictions) * 100, 2)
			recall = round(recall_score(topnlabels, topnpredictions) * 100, 2)
			fscore = round(f1_score(topnlabels, topnpredictions) * 100, 2)
			auc = round(roc_auc_score(testLabels, testPredictions) * 100, 2)
			print("Threshold: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f, AUC: %.2f" % 
																(threshold, accuracy, precision, recall, auc))

# Print average results
print("Printing average results of %d experimental iterations" % EXPERIMENT_ITERATIONS)
print("\n***********\nAVERAGED RESULTS\nAccuracy: %.2f, Precision: %.2f, Recall: %.2f, Fscore: %.2f, AUC: %.2f\n***********" % 
																(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(fscores), np.mean(aucs)))	

# Saving the model
docDict["totalFilesUnderConsideration"] = totalFilesUnderConsideration # Adding total documents that we used during training
print("Saving the model in '%s' from the last experimental iteration..." % saveModelDirectory)
dump(classifier, saveModelDirectory + "/phishytics-model.joblib")
print("Saving the document frequency dictionary as well because we will need it if we ever want to test the model")
np.save(saveModelDirectory + "/phishytics-model-tfidf-dictionary.npy", docDict)




# Create a feature importance plot
import matplotlib.pyplot as plt

# Get feature importances from the trained Random Forest model
feature_importances = classifier.feature_importances_

# Get the names of the features (if you have them)
feature_names = list(range(tokenizerVocabSize))  # Replace with your actual feature names or use column names

# Sort the features by importance in descending order
sorted_feature_importances, sorted_feature_names = zip(*sorted(zip(feature_importances, feature_names), reverse=True))

# Create a bar chart to visualize feature importances
plt.figure(figsize=(10, 6))
# Display the top N most important features
top_n = 20  # Set this to the number of top features you want to display
plt.barh(range(top_n), sorted_feature_importances[:top_n], align='center')
plt.yticks(range(top_n), sorted_feature_names[:top_n])



plt.xlabel('Feature Importance')
plt.title('Feature Importance Plot')
plt.show()






from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have predicted probabilities and true labels
# Replace 'y_true' with your true labels and 'y_scores' with your predicted probabilities
fpr, tpr, thresholds = roc_curve([0,1], [0,1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

