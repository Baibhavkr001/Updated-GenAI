from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)

# Load assessment data
df = pd.read_csv("assessments.csv")

# Vectorize the descriptions for similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["Description"])

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Recommend Endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.get_json()
    query = content.get("query", "")

    if not query:
        return jsonify({"error": "Missing query"}), 400

    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:10]

    recommendations = []
    for idx in top_indices:
        score = cosine_similarities[idx]
        if score > 0:
            assessment = df.iloc[idx]
            recommendations.append({
                "url": assessment["URL"],
                "adaptive_support": assessment["Adaptive_Support"],
                "description": assessment["Description"],
                "duration": int(assessment["Duration"]),
                "remote_support": assessment["Remote_Support"],
                "test_type": eval(assessment["Test_Type"])
            })

    return jsonify({"recommended_assessments": recommendations}), 200

# Frontend
@app.route('/')
def home():
    return render_template("index.html")

# Port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
