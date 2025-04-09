from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("assessments.csv")
df.fillna("", inplace=True)

# Combine all relevant text
df["combined_text"] = df["Assessment_Name"].astype(str) + " " + \
                      df["Description"].astype(str) + " " + \
                      df["Test_Type"].astype(str)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-10:][::-1]

    recommended = []
    for i in top_indices:
        assessment = df.iloc[i]
        recommended.append({
            "url": str(assessment["URL"]),
            "adaptive_support": str(assessment["Adaptive_Support"]),
            "description": str(assessment["Description"]),
            "duration": int(assessment["Duration"]),
            "remote_support": str(assessment["Remote_Support"]),
            "test_type": eval(assessment["Test_Type"]) if isinstance(assessment["Test_Type"], str) else []
        })

    return jsonify({"recommended_assessments": recommended}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
