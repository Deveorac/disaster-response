[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Deveorac/disaster-response">
    <img src="https://c0.wallpaperflare.com/preview/633/410/327/business-communication-concept-illustration.jpg" alt="Logo" width="240" height="200">
  </a>

  <h3 align="center">Twitter Disaster Response Pipeline Project</h3>

  <p align="center">
    Machine learning project for Twitter disaster response. Part of the Udacity Data Scientist Nanodegree program. 
  </p>
</p>




## Table of Contents
1. [Introduction](#introduction)
2. [Running the App](#running)
	1. [Dependencies](#dependencies)
	2. [Installing](#installation)
3. [Acknowledgement](#acknowledgement)
4. [Contact](#contact)

<a name="introduction"></a>
## Description

This project is in collaboration with Udacity and Figure Eight. A dataset containing tweets and messages from disasters was provided with the purpose of developing a Natural Language Processing model to categorize these tweets. 

The project relies on a ETL pipeline and a ML pipeline. A web app also shows the model results. 

<a name="running"></a>
## Running the App

<a name="dependencies"></a>
### Dependencies
* Python 3
* NumPy, SciPy, Pandas, Sciki-Learn
* NLTK
* SQLalchemy
* Pickle
* Flask, Plotly

<a name="installation"></a>
### Installing

1. To set up the database, train model and save the model:

    - ETL pipeline (clean data and store the processed data in the database)
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
    - ML pipeline (load data, train classifier, save the classifier as a pickle file)
        `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Jupyter Notebooks are also provided to walk through the scripts, though these are not required for the app to run. 

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/)
* [Figure Eight](https://www.figure-eight.com/)

<a name="contact"></a>
## Contact

Mehrnaz Siavoshi - [@i_mehrnaz](https://twitter.com/i_mehrnaz)

Project Link: [https://github.com/Deveorac/disaster-response](https://github.com/Deveorac/disaster-response)



![Train Classifier without Category Level Precision Recall](screenshots/train_classifier.png)

6. Sample run of train_classifier.py with precision, recall etc. for each category

![Train Classifier with Category Level Precision Recall](screenshots/train_classifier_category_precision_recall.png)

