import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    将2组数据合并
    :param messages_filepath: 信息数据路径
    :param categories_filepath: 分类数据路径
    :return: 合并数据集 (pandas.DataFrame)
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories,how='left',on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories],axis=1)
    
    
    return df


def clean_data(df):
    """清洗数据：去重"""
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    创建数据库
    :df: 数据集
    :database_filename: 数据库名
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('InsertTableName', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
