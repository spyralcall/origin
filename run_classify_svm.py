from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

files = ['result_kujou_sud/' + path for path in os.listdir('result_kujou_sud')]
count_vect = CountVectorizer(input='filename')
X_train_counts = count_vect.fit_transform(files)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf = X_train_tfidf.toarray()

all_X = X_train_tfidf
all_y = np.loadtxt('only_suuti_label.csv', delimiter = '\n', dtype = float)
print(all_y)
print(type(all_X))
print(type(all_y))


train_X, test_X, train_y, test_y = train_test_split(all_X, all_y,
                                                    test_size=0.2,
                                                    random_state=41)

params = {
	"C": 

print(train_X)
print(test_X)
print(train_y)
print(test_y)

clf2 = SVC(kernel='rbf', C=5, gamma=0.2)

scores = cross_val_score(clf2, all_X, all_y)
print("Cross-Validation scores:{}".format(scores))
print("Average score:{}".format(np.mean(scores)))

clf2.fit(train_X,train_y)
pred_y2 = clf2.predict(test_X)
print(f1_score(test_y, pred_y2, average='macro'))



from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, pred_y2))