from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
class trainmodel:
    def __init__(self,xtrain,ytrain,xtest,ytest):
        self.xtrain=xtrain
        self.ytrain=ytrain
        self.xtest=xtest
        self.ytest=ytest

    def _validate_training_classes(self):
        if len(set(self.ytrain)) < 2:
            raise ValueError("Training data must contain at least 2 classes.")

    def train_Logistic_Regression(self):
        self._validate_training_classes()
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(self.xtrain, self.ytrain)
        return lr_model
    def train_Random_Forest(self):
        self._validate_training_classes()
        rf_model = RandomForestClassifier()
        rf_model.fit(self.xtrain, self.ytrain)
        return rf_model
    def train_Support_Vector_Machine(self):
        self._validate_training_classes()
        svm_model = LinearSVC()
        svm_model.fit(self.xtrain, self.ytrain)
        return svm_model
    def evaluate_model(self,model):
        if model is None:
            return None
        y_pred = model.predict(self.xtest)
        accuracy = accuracy_score(self.ytest, y_pred)
        precision = precision_score(self.ytest, y_pred, zero_division=0)
        recall = recall_score(self.ytest, y_pred, zero_division=0)
        f1 = f1_score(self.ytest, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(self.ytest, y_pred, labels=[0, 1])
        class_report = classification_report(self.ytest, y_pred, zero_division=0)
        return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1, 'conf_matrix':conf_matrix, 'class_report':class_report}
