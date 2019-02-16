import os
import re
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sqlalchemy import create_engine
from joblib import load
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


app = Flask(__name__)



def tokenize(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    clean_tokens = [w for w in clean_tokens if not w in stop_words]
    return clean_tokens


# load data
workspace_path = re.match(r'(.*/).+/.+.py', os.path.realpath(__file__)).group(1)
engine = create_engine('sqlite:///{}data/disaster_response.db'.format(workspace_path))
df = pd.read_sql_table('disaster_response', engine)

# load model
model = load("{}/models/disaster_response.joblib".format(workspace_path))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    message_lengths = df.message.str.len()
    message_lengths = message_lengths[message_lengths < 600]

    list_words = []
    for text in df.message:
        list_words += tokenize(text)
    top_words = pd.Series(list_words).value_counts()
    top_words = top_words[top_words.index.str.len() > 2].iloc[:10]
    top_words_values = top_words.values
    top_words = top_words.index
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words,
                    y=top_words_values
                )
            ],

            'layout': {
                'title': 'Top words in corpus',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=message_lengths,
                    nbinsx=50
                )
            ],

            'layout': {
                'title': 'Distribution of Message length',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Characters"
                }
            }
        }
    ]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
