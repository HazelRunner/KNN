from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import  DecisionTreeClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#加载数据
digits = load_digits()
#print(type(digits))
#<class 'sklearn.utils.Bunch'>
#print(digits.keys())
#dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])
data = digits.data
print(data.shape)
#(1797, 64)
#<class 'numpy.ndarray'>

# print(data[0])          #长度64的一维数组
# print(digits.images[0]) #8*8二维矩阵
# print(digits.target[0]) #数字
# print(digits.target_names[0]) #数字
# print(digits.DESCR[0])        #.
#将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()
#分割数据，30%作为测试集
train_X, test_X, train_y, test_y = train_test_split(data, digits.target, test_size=0.30, random_state=33)
#采用Z-Score做规范化
ss = preprocessing.StandardScaler()
train_ss_X = ss.fit_transform(train_X)
test_ss_X = ss.transform(test_X)
#创建KNN分类器
model = KNeighborsClassifier()
model.fit(train_ss_X, train_y)
predict_y = model.predict(test_ss_X)
print("KNN准确率：", accuracy_score(test_y, predict_y))
#创建SVM分类器
model_svm = svm.SVC()
model_svm.fit(train_ss_X,  train_y)
predict_svm_y = model_svm.predict(test_ss_X)
print("SVM准确率：", accuracy_score(test_y, predict_svm_y))
###############################################
###############################################
#采用Min-Max 规范化
mm = preprocessing.MinMaxScaler()
train_mm_X = mm.fit_transform(train_X)
test_mm_X = mm.transform((test_X))
#创建naive Bayes分类器
mnb = MultinomialNB()
mnb.fit(train_mm_X, train_y)
predict_mnb_y = mnb.predict(test_mm_X)
print("多项式朴素贝叶斯准确率", accuracy_score(test_y, predict_mnb_y))
#创建决策树CART分类器
dtc = DecisionTreeClassifier() #criterion='gini'
dtc.fit(train_mm_X, train_y)
predict_dtc_y = dtc.predict(test_mm_X)
print("CART决策树准确率",accuracy_score(test_y, predict_dtc_y))


'''
SVM准确率： 0.987037037037037
KNN准确率： 0.9796296296296296
多项式朴素贝叶斯准确率 0.8907407407407407
CART决策树准确率 0.8092592592592592

'''
