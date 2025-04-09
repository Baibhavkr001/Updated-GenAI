from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load data
data = pd.read_csv("assessments.csv")

# Precompute TF-IDF vectors on descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['Description'])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    content = request.get_json()
    query = content.get("query", "")

    if not query:
        return jsonify({"error": "Query is missing"}), 400

    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:5]

    recommendations = []
    for i in top_indices:
        rec = {
            "url": data.iloc[i]["URL"],
            "adaptive_support": data.iloc[i]["Adaptive_Support"],
            "description": data.iloc[i]["Description"],
            "duration": int(data.iloc[i]["Duration"]),
            "remote_support": data.iloc[i]["Remote_Support"],
            "test_type": eval(data.iloc[i]["Test_Type"])  # convert string to list
        }
        recommendations.append(rec)

    return jsonify({"recommended_assessments": recommendations})

@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
