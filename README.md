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

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/disaster_response.joblib`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    Note that for run.py to run properly the database needs to be called `disaster_response.db` and 
    the model `disaster_response.joblib`

3. Go to http://0.0.0.0:3001/
