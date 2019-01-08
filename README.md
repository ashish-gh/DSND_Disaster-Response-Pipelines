# Disaster Response Pipeline Project

### Project Overview

The main purpose of this project is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. And it also includes a web app where an emergency worker can input a new message and get classification results in several categories. 



### Project Components

This project contains three main parts: ETL Pipeline, Machine Learning Pipeline and Flask App. 

1. ETL Pipeline
   * **ETL Pipeline Preparation.ipynb**: This jupyter notebook shows the code and development of ETL pipeline.
   * **process_data.py**: This Python script loads the messages and categories datasets, merges the clean data then store the data into a SQLite database. 
2. Machine Line Pipeline
   * **ML Pipeline Preparation.ipynb**: This Jupyther notebook shows the code and develoment of Machine Learning Pipeline.
   * **train_classifier.py**:  This Python script loads the data from a SQLite database. Then it uses the data  to train and tune a Machine Learning model using GridSearchCV.  Finally the model will output as a pickle file. 
3. Flask App
   * The web app can receive a input of new message and returns classification results in several categories. 



### Ruinning Instructions

1. Run the following commands in the project's root directory to set up the database and model.
   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run the web app.
   `python run.py`
3. Go to http://0.0.0.0:3001/



### Screenshots of the web app

![image-20190108135021589](https://ws2.sinaimg.cn/large/006tNc79gy1fyz4qbvm0tj31o40u00v5.jpg)

![image-20190108134950399](https://ws2.sinaimg.cn/large/006tNc79gy1fyz4pu37anj31si0o278d.jpg)

![img](https://ws2.sinaimg.cn/large/006tNc79gy1fyxt8ftmy5j30uk0p0wgm.jpg)

