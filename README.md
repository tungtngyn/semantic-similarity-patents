## Semantic Similarity Predictions from Phrases in U.S. Patents

### Project Description
The goal of this project is to predict a semantic similarty score between an anchor phrase and a target phrase, both taken from the text of miscellaneous U.S. Patents. The dataset comes from [this Kaggle repository](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) and comprises of ~35K anchor-target phrase pairs which have been labeled by domain-knowledge experts with a similarity score ranging from 0.0 (no relation between the phrases) to 1.0 (phrases are very similar). The scores provided are discrete, with the only possible values being [0.00, 0.25, 0.50, 0.75, 1.00]. Thus, this can both be a regression and/or classification problem depending on the use-case of the model.

The primary learning objective of this project was to become familiar with both traditional and modern approaches to natural language processing and text feature engineering.


### Technologies & Methods Used
* scikit-learn for misc. ML utilities
* pandas & numpy for data analysis
* nltk for text preprocessing utilities
* spaCy for pre-trained word vectors
* gensim for training word2vec and FastText models from scratch
* huggingface's transformer & datasets library for a pre-trained BERT model
* Google Colab & TensorFlow for BERT model training


### Project Directory
./exploratory-data-analysis.ipynb
* Preliminary exploration of the dataset.

./linear_models.ipynb
* Trained linear and logistic models on various text feature engineering approaches
* Tested Levenshtein distance (edit distance), pre-trained word vectors, and custom-trained word vectors

./train-patents-embeddings.ipynb
* Trained a word2vec and FastText model on text from U.S. Patents filed between Jan. 2022 - Jul. 2022
* Due to resource constraints, only 6mo. worth of patents could be processed even though ~50 years of data was available
    * All training was done locally and gensim does not support GPU acceleration. 6mo took ~24 hrs of runtime!

./advanced_models.ipynb
* Tuned a Random Forest using a combination of RandomizedSearchCV and GridSearchCV
* Fine-tuned a pre-trained BERT model using TensorFlow

./contractions.py
* File containing a dictionary of common contractions in english
* Used in train-patents-embeddings.ipynb as part of the text preprocessing script

./results/
* ROC, PR curves for all models


### Summary of Results
* Both Regression and Classification approaches were taken as part of this project.

* Key take-aways from exploratory data analysis:
    * The phrases themselves are relatively short (2 +/- 1 words). Majority of phrase pairs do not have any words in common. 

    * Target phrases have much more variety than anchor phrases (which act almost like 'categories'); ~9K unique words, with most target phrases being unique in the entire training dataset (~29K/36K are unique phrases)

    * Phrase pairs with words in common tend to have, on average, a higher similarity score, though there exist contradictions to this generalization (e.g. phrase pairs with alot of words in common but low similarity score).

    * Dataset spans many domains; 106 patent classification codes present

    * Dataset is skewed toward phrase pairs that are unsimilar (e.g. low similarity scores). A OneVsRest scheme is also used. AUPR was the primary metric used for evaluation of model accuracy.

* Multiple feature engineering methods were explored, such as:
    * Levenshtein distance (e.g. edit distance) b/t anchor & target phrase as the feature

    * Various permutations of pre-trained spaCy word vectors as the features (avg, multiply, cosine similarity, etc.)

    * Various permutations of custom-trained word vectors as the features (Refer to: train-patents-embeddings.ipynb)    

* All 1D feature engineering methods, such as cosine similarity and Levenshtein distance did not exhibit any trends. Some models had a negative R^2 value, which indicate the fit does not follow the trend. For advanced models (e.g. Random Forest), only multidimensional feature engineering methods were explored (e.g. spaCy word embeddings)

* The best performing Regressor from this project was the Random Forest tuned via RandomizedSearchCV and GridSearchCV, with an RMSE of ~0.2, e.g. the model is off by 20% on average (score ranges from 0% to 100%).

* The best performing Classifier from this project was also the Random Forest model. The pre-trained base BERT model (12 attention heads) underperformed the Random Forest for 4/5 labels with respects to AUPR. However, the BERT model was only trained for a few epochs. 


### Areas for Improvement & Further Experimentation

* One area of improvement is in the embeddings used for this project. Pre-trained word vectors & BERT models were used, however, these may not be suitable for problems relating to technical documentation (which contain a large amount of uncommon english words and domain-specific acronyms). The performance of the model may improve if resources were available to take advantage of the ~50 yrs of U.S. Patents text available online.

* Hyperparameter tuning was performed using RandomizedSearchCV and GridSearchCV (a basic coarse to fine approach) with limited parameter ranges due to the computational resources and time available. An improvement would be to use a more robust, less computationally intensive hyperparameter tuning methodology, such as Bayesian Optimization and/or Evolutionary Optimization, to squeeze out a bit more performance. Instead of a blind, arbitrary GridSearch, one can also run a Design of Experiments to more efficiently fill in the parameter space. 

* A base BERT model was used in this project. As with most ML problems, scaling up will most likely improve accuracy. More attention heads!

* Different lemmatizers and stemmers were not investigated. Optimizing the lemmatizer and/or stemmer could also have a big impact on model accuracy.