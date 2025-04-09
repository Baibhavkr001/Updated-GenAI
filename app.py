from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv("assessments.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Vectorize descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.get_json()
    user_query = content.get("query", "")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    user_vec = vectorizer.transform([user_query])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[::-1][:3]

    results = []
    for idx in top_indices:
        results.append({
            "Assessment Name": df.iloc[idx]["Assessment_Name"],
            "URL": df.iloc[idx]["URL"],
            "Duration": df.iloc[idx]["Duration"],
            "Adaptive Support": df.iloc[idx]["Adaptive_Support"],
            "Remote Support": df.iloc[idx]["Remote_Support"],
            "Test Type": df.iloc[idx]["Test_Type"]
        })

    return jsonify(results)

# ✅ Render-specific PORT binding
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT
    app.run(debug=True, host="0.0.0.0", port=port)
