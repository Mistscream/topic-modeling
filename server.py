from flask import Flask, jsonify, request
from search.searches import fulltext_search, lda_search, lsi_search
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def root():
    search_mode = 'lda' if not request.json['mode'] else request.json['mode']
    query = '' if not request.json['query'] else request.json['query']
    max_results = 25 if not request.json['maxResults'] else int(request.json['maxResults'])

    search = {
        'lda': lda_search,
        'lsi': lsi_search,
        'fulltext': fulltext_search
    }

    response = {
        'data': {
            'search_mode': search_mode,
            'query': query,
            'max_results': max_results,
            'results': search[search_mode](query, max_results)
        }
    }
        
    return jsonify(response)
