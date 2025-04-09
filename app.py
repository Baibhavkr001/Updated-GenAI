from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)

# Load CSV file
df = pd.read_csv('assessments.csv')

# Use the correct column names from your CSV
df['combined_text'] = df['Assessment_Name'].fillna('') + ' ' + df['Description'].fillna('')

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

    # Cosine similarity
    similarity_scores = cosine_similarity(job_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:10]  # top 10

    recommended = df.iloc[top_indices][[
        'Title', 'Description', 'Duration', 'Adaptive_Support', 'Remote_Support', 'URL'
    ]]

    recommendations = recommended.to_dict(orient='records')
    return jsonify({"recommendations": recommendations})

# Run the app (required for Render)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
