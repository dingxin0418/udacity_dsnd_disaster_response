import sys
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    读取数据
    :database_filepath: 数据库路径
    :return: X (分词信息), y(标签信息), catgory_names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from InsertTableName where related is not null ', engine)  
    X = df.message.values
    y = df.iloc[:,4:].values
    category_names=df.columns[4:].values
   
    return X, y, category_names

def tokenize(text):
    """
    分词
    :text:信息数据
    :return: 分词结果列表
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """搭建模型管道"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {

            'clf__estimator__n_estimators': [50,100],
            'clf__estimator__min_samples_split': [2,5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    模型评估
    :model: 分类模型
    :X_test: 测试数据
    :Y_test: 测试标签
    :category_names:分类数据
    :return:模型结果报告
    """
    y_pred = model.predict(X_test)

    for i in range(36):
        print('category:'+category_names[i])
        print(classification_report(Y_test.T[i], y_pred.T[i]))

def save_model(model, model_filepath):
    """
    储存模型
    :model: 模型
    :model_filepath: pkl文件储存路径
    """
    with open(model_filepath, 'wb') as fid:
        pickle.dump(model, fid)  


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
