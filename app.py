from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)

# Load the CSV file
df = pd.read_csv('assessments.csv')

# Combine title and description for vectorization
df['combined_text'] = df['Title'].fillna('') + ' ' + df['Description'].fillna('')

@app.route("/", methods=["GET"])
def home():
    return "Assessment Recommender is running!"

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"})

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    job_description = data.get("job_description", "")

    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    job_vec = vectorizer.transform([job_description])

    # Cosine similarity to find top 10
    similarity_scores = cosine_similarity(job_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:4]

    recommended = df.iloc[top_indices][[
        'Title', 'Description', 'Duration', 'Adaptive_Support', 'Remote_Support', 'Test_Type', 'URL'
    ]]

    # Convert to list of dicts for JSON
    recommendations = []
    for _, row in recommended.iterrows():
        recommendations.append({
            "title": row["Title"],
            "description": row["Description"],
            "duration": int(row["Duration"]),
            "adaptive_support": row["Adaptive_Support"],
            "remote_support": row["Remote_Support"],
            "test_type": eval(row["Test_Type"]) if isinstance(row["Test_Type"], str) else [],
            "url": row["URL"]
        })

    return jsonify({"recommendations": recommendations})

# Run the app with port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
