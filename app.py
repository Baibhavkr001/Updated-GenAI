from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv("assessments.csv")

# Lowercase assessment names for comparison
df["Assessment_Name_lower"] = df["Assessment_Name"].str.lower()

# Vectorize using description
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["Description"])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get("query", "").strip().lower()

    if query == "":
        return jsonify({"error": "Query is missing"}), 400

    if query in df["Assessment_Name_lower"].values:
        idx = df[df["Assessment_Name_lower"] == query].index[0]
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        similar_indices = sim_scores.argsort()[::-1][1:3]
        recommended = df.iloc[similar_indices]
    else:
        return jsonify({"message": "No match found for given assessment name"}), 200

    output = []
    for _, row in recommended.iterrows():
        output.append({
            "url": row["URL"],
            "adaptive_support": row["Adaptive_Support"],
            "description": row["Description"],
            "duration": int(row["Duration"]),
            "remote_support": row["Remote_Support"],
            "test_type": eval(row["Test_Type"]) if isinstance(row["Test_Type"], str) else []
        })

    return jsonify({"recommended_assessments": output}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
