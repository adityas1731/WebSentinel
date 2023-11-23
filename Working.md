A high-level explanation of how the three provided Python scripts work together as part of a larger workflow for phishing detection:

#### 1. Data Tokenization

1. **`create_data_for_tokenization.py`**:

   - This script is responsible for preparing the data for training the phishing detection model.
   - It collects HTML files from two subdirectories: "legitimate_htmls" and "phishing_htmls" within the labeled data folder.
   - It tokenizes the HTML content using Byte-Level Byte-Pair Encoding (BPE).
   - TF-IDF scores are computed for tokens based on document frequency.
   - The script then trains a Random Forest classifier using the TF-IDF-weighted tokens and associated labels (0 for legitimate, 1 for phishing).
   
   - The trained model is saved to a specified directory along with the document frequency dictionary for future use.
   
   
#### 2. Model Training

2. **`train_phishing_detection_model.py`**:


->**Argument Parsing**:
   - The script starts by parsing command-line arguments to configure various settings and options, including the location of tokenizer files, labeled data, language filtering, and threshold testing.

->**Data Preparation**:
   - It loads tokenization files (tokenizer vocabulary and merges) and the document frequency dictionary required for training and evaluation.

-> **Data Splitting**:
   - The script splits the data into training and testing (validation) sets. This separation is essential for assessing the model's performance on unseen data.

->**Model Training**:
   - A Random Forest classifier is trained on the training data. Random Forest is a machine learning algorithm that can handle both classification and regression tasks.

->**Model Evaluation**:
   - The trained model is evaluated using various metrics to assess its performance. These metrics include accuracy, precision, recall, F1-score, and AUC (Area Under the Curve). These metrics help gauge how well the model can classify websites as legitimate or phishing.

->**Threshold Testing** (Optional):
   - The script optionally allows for testing different confidence thresholds. By adjusting the threshold, you can explore how it affects the model's performance. This step is useful for fine-tuning the model and understanding the trade-offs between false positives and false negatives. *By adjusting the threshold, you can control the trade-off between false positives and false negatives. A lower threshold may result in more websites being classified as "PHISHING,"*

->**Results Reporting**:
   - The script prints the average results of multiple experimental iterations. This provides a summary of the model's overall performance.

->**Model Saving**:
   - The trained model is saved to a specified directory. Additionally, the document frequency dictionary, which is necessary for testing the model, is also saved. Saving the model allows you to use it for future phishing detection tasks.

In essence, the script covers the entire process of training and evaluating a phishing detection model, including data preparation, model training, and assessment of its performance using various metrics. The optional threshold testing and saving of the trained model enhance its utility and flexibility.

#### 3. Model Testing

3. **`test_model.py`**:

   - This script allows you to test the trained phishing detection model on a specific website.
   - It takes command-line arguments to specify the location of tokenizer files, the threshold for classification, the directory of the trained model, and the website to test.
   - The script loads the tokenizer, model, and document frequency dictionary.
   - It retrieves the HTML content of the specified website using web requests.
   - The HTML content is tokenized and converted into a TF-IDF-weighted feature vector.
   - The model then **predicts** the probability of the website being phishing.
   - The user-defined threshold is applied to classify the website as "PHISHING" or "NOT PHISHING."
   - The script displays the prediction probability and the final classification based on the threshold.

Together, these scripts create a workflow for training, testing, and evaluating a phishing detection model. The first script prepares the data and trains the model, the second script trains and evaluates the model with various options, and the third script allows users to apply the trained model to new websites and determine their phishing status. The threshold in the third script is a user-configurable parameter that influences the binary classification decision.