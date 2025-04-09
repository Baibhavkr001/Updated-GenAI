from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

# Load the dataset
df = pd.read_csv("assessments.csv")

# Fill NaNs with empty string
df.fillna("", inplace=True)

# Combine all relevant text for better recommendations
df["combined_text"] = df["Assessment_Name"].astype(str) + " " + \
                      df["Description"].astype(str) + " " + \
                      df["Test_Type"].astype(str)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    job_desc = data.get("job_description", "")

    if not job_desc:
        return jsonify({"error": "Job description cannot be empty"}), 400

    query_vec = vectorizer.transform([job_desc])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-3:][::-1]  # Top 3 recommendations

    recommendations = []
    for i in top_indices:
        rec = {
            "Assessment_Name": str(df.iloc[i]["Assessment_Name"]),
            "Description": str(df.iloc[i]["Description"]),
            "Duration": int(df.iloc[i]["Duration"]),
            "Adaptive_Support": str(df.iloc[i]["Adaptive_Support"]),
            "Remote_Support": str(df.iloc[i]["Remote_Support"]),
            "Test_Type": str(df.iloc[i]["Test_Type"]),
            "URL": str(df.iloc[i]["URL"])
        }
        recommendations.append(rec)

    return jsonify(recommendations)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
