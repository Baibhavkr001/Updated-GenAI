from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import os

# Load data
df = pd.read_csv("genai_assessments.csv")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Vectorize descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["Description"])

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.get_json()
    query = content.get("query", "").strip()

    if query == "":
        return jsonify({"error": "Query cannot be empty"}), 400

    if query in df["Assessment_Name"].values:
        idx = df[df["Assessment_Name"] == query].index[0]
        cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[::-1][1:3]  # top 2 excluding self

        recommendations = df.iloc[similar_indices].to_dict(orient="records")

        result = []
        for rec in recommendations:
            result.append({
                "url": rec["URL"],
                "adaptive_support": rec["Adaptive_Support"],
                "description": rec["Description"],
                "duration": int(rec["Duration"]),
                "remote_support": rec["Remote_Support"],
                "test_type": eval(rec["Test_Type"]) if isinstance(rec["Test_Type"], str) else rec["Test_Type"]
            })

        return jsonify({"recommended_assessments": result}), 200

    return jsonify({"recommended_assessments": []}), 200

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
