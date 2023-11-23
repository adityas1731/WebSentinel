	-> Install python and pip library.
 -> Install rust from its official website 
 
**Windows**
-> Install numpy : 
	pip install numpy

-> Install pandas
	pip install pandas

-> Install joblib
	pip install joblib

-> Install tokenizers
	pip install tokenizer
	
-> Install langdetect
	pip install langdetect

-> Install scikit-learn
	pip install scikit-learn

-> Install tkinter
	pip install tk
	
```
pip install numpy pandas joblib tokenizers langdetect scikit-learn tk
```



**Linux**

Run ./install_packages.sh in terminal to install the required packages:
	numpy
	pandas
	joblib
	tokenizers
	angdetect
	scikit-learn
	tkinter

#### 1. Data Tokenization

**(create_data_for_tokenization.py)**

In order to train a phishing website detection model, you first need to tokenize all the HTML files into tokens using Byte Pair Encoding (BPE). We will use the tokenizer library for this. Once the html files are in their respective folders, run the following command.

```

python create_data_for_tokenization.py --labeled_data_folder labeled_data --vocab_size 300 --min_frequency 5

```

The script takes three parameters as inputs:

- labeled_data_folder: Folder containing data for phishing and legitimate websites.

- vocab_size: Maximum number of tokens to have in the vocabulary

- min_frequency: Tokens having frequency lower than this value will be ignored

This script is designed for preprocessing HTML data, tokenizing it using Byte-Level BPE, and saving the tokenizer's vocabulary and configuration for further use.

  

#### 2. Model Training

**(train_phishing_detection_model.py)**

Once we have create a Byte Pair Encoding tokenizer, we will be able to use it to tokenize HTML files and extract features for machine learning. On top of BPE tokens, we will apply TFIDF scores to get a feature representation of each HTML file. Run the following command to train your own model.

```

python train_phishing_detection_model.py --tokenizer_folder tokenizer/ --labeled_data_folder labeled_data/ --ignore_other_languages 1 --apply_different_thresholds 1 --save_model_dir saved_models

```

The script takes five parameters as inputs:

- tokenizer_folder: Folder containing tokenizer files. The default folder is 'tokenizers'

- labeled_data_folder: Folder containing data for phishing and legitimate websites.

- ignore_other_languages: Whether to ignore languages other than english. Set it to 0 if you want to include all languages.

- apply_different_thresholds: Whether to apply different confidence thresholds during model evaluation.

- save_model_dir: Directory to save to model files

  

#### 3. Model Testing

Once we have a trained model, we can simply test it live on any website using the following command.

**(test_model.py)**

```

python test_model.py --tokenizer_folder tokenizer --threshold 0.5 --model_dir saved_models --website_to_test *url*

```

The script takes four parameters as inputs:

- tokenizer_folder: Folder containing tokenizer files. The default folder is 'tokenizers'

- threshold: Threshold to use for making final predictions. By default, the value is 0.5.

- model_dir: Directory where saved model files exist.

- website_to_test: Website you want to test. Please add "http://" or "https://" before the website to make everything work. Otherwise, you will face an error.

  

### Using Pre-trained Model

To use the pre-trained model, **please go to the 'pretrained_models' directory and unzip the 'document-frequency-dictionary.zip' file**. Do not unzip it in a new directory, keep it in the same directory. Once that is done, you can run the following command to use the pre-trained model.

```

python test_pretrained_model.py --tokenizer_folder pretrained_models --threshold 0.5 --model_dir pretrained_models --website_to_test *url*

```

The script takes four parameters as inputs:

- tokenizer_folder: Folder containing tokenizer files. The default folder is 'tokenizers' but here we will use 'pretrained_models'.

- threshold: Threshold to use for making final predictions. By default, the value is 0.5.

- model_dir: Directory where saved model files exist. The pre-trained model files exist in 'pretrained_models'.

- website_to_test: Website you want to test. Please add "http://" or "https://" before the website to make everything work. Otherwise, you will face an error.



************************************************************************

