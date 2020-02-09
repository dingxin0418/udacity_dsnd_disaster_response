# 灾害因对通道
## 里头有啥
1. etl管道文件process_data.py，用于清洗数据，message数据打标分类存入数据库用于后续模型训练
2. nlp管道train_classifier.py，使用`tokenize()`进行分词，tfidf词向量化处理后，进行模型训练
3. run.py运行程序，本地生成html包含可视化工具和信息预测
## 如何使用
1. 在根目录下运行下列指令建立你的模型和数据库
    - 执行ETL pipeline 做数据清洗，并储存于数据库中
    
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/sqlite:///DisasterResponse.db`
    - 执行nlp管道输出模型
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. 执行
    `python run.py`

3. 点开此链接 http://0.0.0.0:3001/
