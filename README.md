# Disaster Response Pipeline Project
This project will takes message data stored in csv files and process it using an ETL and ML pipeline. The result will then
be accessible through a webapp running locally. The data has been provided by [figure-eight](https://www.figure-eight.com/)
and it is comprised of real messages that were sent during disaster events, 
that have been categorized based on the nature of the message: e.g. aid, earthquake, shelter, water, etc.
The application will determine to which category any new message belongs to.

### Packages:
This project was developed using:
* pandas 0.24.1 
* sqlalchemy 1.2.17 
* nltk 3.4
* scikit-learn 0.20.2
* plotly 3.5.0
* flask 1.0.2
* joblib 0.13.1
### Files:
The project has the following file structure:
```
workspace
├── README.md
├── app
|   ├── run.py
|   └── templates
|       ├──go.html
|       └──master.html
├── data
|   ├── disaster_categories.csv
|   ├── disaster_messages.csv
|   └── process_data.py
└── models
    └── train_classifier.py

```
* __app__: Folder containing the webapp. Html templates are stored in the subfolder `templates`. running the `run.py` file 
will start the flask app.
* __data__: Folder containing the raw data in csv file form as well as the `process_data.py` file which starts an ETL pipeline
generating a sqlite database.
* __models__: Folder containing `train_classifier.py` which trains and saves a model to be used in the app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_response.db models/disaster_response`

2. Run the following command in the app's directory to run your web app.
    `python run.py`.
    
    Note that for run.py to run properly the database needs to be called `disaster_response.db` and 
    the model `disaster_response.joblib`

3. Go to http://0.0.0.0:3001/
