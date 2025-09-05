# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from wordcloud import WordCloud
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# # Importing ML Models
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import accuracy_score
# import warnings


# warnings.filterwarnings('ignore')
# nltk.download('punkt')


# data_path = r"C:\Users\Hp\Desktop\Recommendation Systems\Data\files (4).tsv"
# data=pd.read_csv(data_path, sep="\t")

# data.head()

# data.shape

# data.isnull().sum()

# data['Liked'].value_counts()

# # Adding Columns of char, word, and sent count
# # No. of Characters column
# data['char_count']=data['Review'].apply(len)
# # No. of Words count column
# data['word_count']=data['Review'].apply(lambda x :len(str(x).split()))
# # No. of Sent count column
# data['sent_count']=data['Review'].apply(lambda x:len(nltk.sent_tokenize(str(x))))

# data.head()

# # avg #char in positive reviews
# data[data['Liked']==1]['char_count'].mean()

# # avg #char in negative reviews
# data[data['Liked']==0]['char_count'].mean()

# data['Review'][1]

# review=re.sub('[^a-zA-Z]',' ',data['Review'][1])

# review=review.lower()

# review=review.split()

# review

# all_stopwords=stopwords.words('english')
# all_stopwords.remove('not')

# all_stopwords

# review=[word for word in review if word not in set(all_stopwords)]

# review

# custom_stopwords = {
#     'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
#     'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
#     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
#     'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
#     'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
# }
# corpus=[]
# ps = PorterStemmer()
# stop_words=set(stopwords.words("english"))-custom_stopwords

# for i in range(len(data)):
#     review=re.sub('[^a-zA-Z]',' ',data['Review'][i])
#     review=review.lower()
#     review=review.split()
#     review=[ps.stem(word) for word in review if word not in stop_words]
#     review=" ".join(review)
#     corpus.append(review)


# data['processed_text']=corpus


# data.head()

# wc=WordCloud(width=500, height=500,min_font_size=8, background_color='white')

# pos_reviews = wc.generate(data[data['Liked'] == 1]['processed_text'].str.cat(sep=" "))

# plt.imshow(pos_reviews)

# neg_reviews = wc.generate(data[data['Liked'] == 0]['processed_text'].str.cat(sep=" "))

# plt.imshow(neg_reviews)



# cv=CountVectorizer(max_features=1500)
# X=cv.fit_transform(corpus).toarray()
# X


# X.shape

# y=data['Liked']

# y


# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=42)

# import pandas as pd

# # Check where y_train is NaN
# nan_rows = X_train[pd.isna(y_train)]

# # Print the corresponding X_train rows and y_train values
# print("Rows in X_train corresponding to NaN in y_train:")
# print(nan_rows)

# print("\nNaN values in y_train:")
# print(y_train[pd.isna(y_train)])


# # Training Naive Bayes Model

# nb=GaussianNB()
# nb.fit(X_train, y_train)
# y_pred_nb=nb.predict(X_test)

# accuracy_score(y_test,y_pred_nb)
# # Import libraries
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score, confusion_matrix
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt


# # Confusion Matrix
# cm_lr = confusion_matrix(y_test, y_pred_nb)
# sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens")
# plt.title("Gaussian NB Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # Training Logistic Regression Model
# lr=LogisticRegression()
# lr.fit(X_train, y_train)
# y_pred_lr=lr.predict(X_test)
# accuracy_score(y_test,y_pred_lr)

# # Import libraries

# from sklearn.metrics import accuracy_score, confusion_matrix
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt


# # Confusion Matrix
# cm_lr = confusion_matrix(y_test, y_pred_lr)
# sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens")
# plt.title("Logistic Regression Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()


# # Import libraries
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.calibration import CalibratedClassifierCV
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # -------------------------------
# # RANDOM FOREST MODEL
# # -------------------------------
# # Initialize Random Forest
# rf = RandomForestClassifier(random_state=42)

# # Hyperparameter grid for RandomizedSearchCV
# rf_param_dist = {
#     'n_estimators': [50, 100, 200, 500],
#     'max_depth': [None, 5, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# # Randomized search for RF
# rf_search = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=rf_param_dist,
#     n_iter=50,
#     cv=5,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )

# # Fit RF
# rf_search.fit(X_train, y_train)

# # Predictions
# y_pred_rf = rf_search.predict(X_test)

# # Accuracy
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print("Random Forest Best Parameters:", rf_search.best_params_)
# print("Random Forest Test Accuracy:", accuracy_rf)

# # Confusion Matrix
# cm_rf = confusion_matrix(y_test, y_pred_rf)
# sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues")
# plt.title("Random Forest Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()



# # Didnt use the **"poly"** kernel as it takes a lot of computation and RAM memory. Tried the kernel but the best kernel output was from **"rbf"**. 

# # Import libraries
# from sklearn.svm import SVC
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Define parameter grid for randomized search
# param_dist = {
#     'C': np.logspace(-3, 3, 7),          # 0.001 to 1000
#     'gamma': np.logspace(-3, 3, 7),      # 0.001 to 1000
#     'kernel': ['linear', 'rbf']  # Kernel types
# }

# # Initialize SVM
# svc = SVC()

# # Configure RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=svc,
#     param_distributions=param_dist,
#     n_iter=100,      # Number of random combinations
#     cv=5,            # 5-fold cross-validation
#     verbose=2,
#     random_state=42,
#     n_jobs=-1        # Use all CPU cores
# )

# # Fit model on training data
# random_search.fit(X_train, y_train)

# # Get best parameters and score
# best_params = random_search.best_params_
# best_cv_score = random_search.best_score_

# # Predict on test data
# y_pred = random_search.predict(X_test)

# # Evaluate accuracy
# test_accuracy = accuracy_score(y_test, y_pred)

# # Print results
# print("Best Parameters:", best_params)
# print("Best Cross-Validation Score:", best_cv_score)
# print("Test Set Accuracy:", test_accuracy)


# # Confusion Matrix
# cm_svm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens")
# plt.title("SVM Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()


# # From below metric results, we conclude that Logistic Regression and Support Vector Machine (SVM) has similar better Benchmark Metrics, with highest recall and precision on both negative and positive sentiments.

# # -------------------------------
# # IMPORTS
# # -------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

# # -------------------------------
# # RANDOM FOREST
# # -------------------------------
# rf = RandomForestClassifier(random_state=42)
# rf_param_dist = {
#     'n_estimators': [50, 100, 200, 500],
#     'max_depth': [None, 5, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# rf_search = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=rf_param_dist,
#     n_iter=50,
#     cv=5,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )
# rf_search.fit(X_train, y_train)
# y_pred_rf = rf_search.predict(X_test)
# y_proba_rf = rf_search.predict_proba(X_test)[:,1]

# # -------------------------------
# # SVM
# # -------------------------------
# svc = SVC(probability=True)
# svm_param_dist = {
#     'C': np.logspace(-3,3,7),
#     'gamma': np.logspace(-3,3,7),
#     'kernel': ['linear','rbf']
# }
# svm_search = RandomizedSearchCV(svc, svm_param_dist, n_iter=50, cv=5, verbose=1, n_jobs=-1, random_state=42)
# svm_search.fit(X_train, y_train)
# calibrated_svc = CalibratedClassifierCV(estimator=svm_search.best_estimator_, cv=5)
# calibrated_svc.fit(X_train, y_train)
# y_pred_svm = calibrated_svc.predict(X_test)
# y_proba_svm = calibrated_svc.predict_proba(X_test)[:,1]

# # -------------------------------
# # NAIVE BAYES
# # -------------------------------
# nb = GaussianNB()
# nb.fit(X_train, y_train)
# y_pred_nb = nb.predict(X_test)
# y_proba_nb = nb.predict_proba(X_test)[:,1]

# # -------------------------------
# # LOGISTIC REGRESSION
# # -------------------------------
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)
# y_pred_lr = lr.predict(X_test)
# y_proba_lr = lr.predict_proba(X_test)[:,1]

# # -------------------------------
# # CONFUSION MATRICES
# # -------------------------------
# models = {'Random Forest': y_pred_rf, 'SVM': y_pred_svm, 'Naive Bayes': y_pred_nb, 'Logistic Regression': y_pred_lr}
# for name, pred in models.items():
#     cm = confusion_matrix(y_test, pred)
#     plt.figure(figsize=(5,4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'{name} Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()

# # -------------------------------
# # ROC CURVE AND AUC
# # -------------------------------
# plt.figure(figsize=(10,7))

# fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
# auc_rf = auc(fpr_rf, tpr_rf)
# plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.2f})')

# fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
# auc_svm = auc(fpr_svm, tpr_svm)
# plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC={auc_svm:.2f})')

# fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
# auc_nb = auc(fpr_nb, tpr_nb)
# plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC={auc_nb:.2f})')

# fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
# auc_lr = auc(fpr_lr, tpr_lr)
# plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.2f})')

# plt.plot([0,1], [0,1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves for All Models')
# plt.legend()
# plt.show()

# # -------------------------------
# # CLASSIFICATION REPORTS
# # -------------------------------
# print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
# print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
# print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))
# print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))


# # **Conclusion:** From above accuracy results, we conclude that Support Vector Machine (SVM) performs best on the data.

# import joblib
# import shutil
# from IPython.display import FileLink


# # Train the model
# svc = SVC(kernel='rbf', gamma=0.1, C=1.0)
# svc.fit(X_train, y_train)

# # Save the model
# joblib.dump(svc, "Restaurant_review_model.joblib")


# # **Best Parameters: {'kernel': 'rbf', 'gamma': np.float64(0.1), 'C': np.float64(1.0)}**.
# # Parameters are based on results from multiple kernel iterations executed above.

# # Save the CountVectorizer
# joblib.dump(cv, "count_v_res.joblib")


# **Inferences in the Model** : Predictions and Results
# For Prediction result, Please run this block below, Models are already trained and saved in **saved_model** directory.

import joblib
import re

# Load trained model and vectorizer
svc = joblib.load("saved_model\\Restaurant_review_model.joblib")
cv = joblib.load("saved_model\\count_v_res.joblib")

# Example input reviews
new_reviews = [
    "The chomein as bland and tasteless. There were no seasonings and sausage. Worst food ever.",
    "Service was bad, although food was not good.",
    "Momo was great! Momo was not good."
]

# Stop characters for splitting
stop_list = r'[.!;]'

for review in new_reviews:
    # --- Split review into sub-sentences ---
    sub_sentences = re.split(stop_list, review)
    sub_sentences = [s.strip() for s in sub_sentences if s.strip()]  # remove empty parts

    # --- Predict each sub-sentence ---
    sent_preds = []
    for sub in sub_sentences:
        X_sub = cv.transform([sub]).toarray()
        pred = svc.predict(X_sub)[0]  # 1 = Positive, 0 = Negative
        sent_preds.append(pred)

    # --- Apply logic ---
    if all(p == 1 for p in sent_preds):
        final_p = 1  # Positive
    elif all(p == 0 for p in sent_preds):
        final_p = 0  # Negative
    else:
        final_p = 2  # Mixed

    # Map numeric sentiment to string for printing
    sentiment_map = {0: "Negative", 1: "Positive", 2: "Mixed"}
    review_sentiment = sentiment_map[final_p]

    # --- Output ---
    print(f"Review: {review}")
    # print(f"Sub-sentences: {sub_sentences}") run only for intuition purpose
    sent_preds = [0, 1, 1, 0]
    pred_labels = [sentiment_map[p] for p in sent_preds]
    # print(f"Predictions: {pred_labels}") this too uncomment only for intuition purpose
    print(f"Final Sentiment: {review_sentiment} (p={final_p})\n")


