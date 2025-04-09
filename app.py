from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load assessments
df = pd.read_csv('assessments.csv')

# Lowercase titles for search
df['Title_lower'] = df['Title'].str.lower()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    job_desc = data.get('job_description', '').lower()

    # Find matching assessments
    matched = df[df['Title_lower'].str.contains(job_desc)]

    if matched.empty:
        return jsonify({'error': 'Assessment not found'}), 404

    results = []
    for _, row in matched.iterrows():
        results.append({
            'title': row['Title'],
            'description': row['Description'],
            'duration': int(row['Duration']),
            'adaptive_support': row['Adaptive_Support'],
            'remote_support': row['Remote_Support'],
            'test_type': eval(row['Test_Type']) if isinstance(row['Test_Type'], str) else [],
            'url': row['URL']
        })

    return jsonify({'assessments': results})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # For Render deployment
    app.run(host='0.0.0.0', port=port)
