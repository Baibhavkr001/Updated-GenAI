from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)

# Load the CSV
df = pd.read_csv("assessments.csv")

# Vectorize using 'Description' column
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the GenAI Assessment Recommendation API!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_input = data.get("query", "")

    if not user_input:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarity.argsort()[-5:][::-1]
    recommendations = df.iloc[top_indices]

    result = []
    for _, row in recommendations.iterrows():
        result.append({
            "assessment_name": row["Title"],
            "description": row["Description"],
            "duration": row["Duration"],
            "adaptive_support": row["Adaptive_Support"],
            "remote_support": row["Remote_Support"],
            "test_type": row["Test_Type"],
            "url": row["URL"]
        })

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
