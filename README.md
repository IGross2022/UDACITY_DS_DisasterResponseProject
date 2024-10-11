# Disaster Response Pipeline Project
Second Project for UDACITY Data Science Nanodegree

''''''''''''''''''''''''''''''''''''''''''''''''''

Project szenario:

*data/process_data.py* uses 2 files (*data/disaster_categories.csv* and *data/disaster_messages.csv*), merges data from both files together, cleans the data and saves it in SQLLite database *data/DisasterResponse.db* (Table *DisasterResponseTable*).

*models/train_classifier.py* reads cleaned data from the SQLLite database *data/DisasterResponse.db* (Table *DisasterResponseTable*) and creates a ML moldel (*models/classifier.pkl*) using RandomForestClassifier.

*app/run.py* starts a WebApplication using the model created abough as well as both .html files in *templates* directory.
The WebApplication acceps disaster messages and using the model predictions assigns them to the respective categories.

''''''''''''''''''''''''''''''''''''''''''''''''''



## Instructions:

ATTENTION:

Before start check the scikit-learn version of your environment (it should be not older than 0.24) and upgrade if necessary:

pip show scikit-learn

pip install --upgrade scikit-learn

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
