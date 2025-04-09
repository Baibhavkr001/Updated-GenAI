from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the data
df = pd.read_csv("Assessments.csv")

# Fill NaN values in 'Title' column
df['Title'] = df['Title'].fillna('')

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Title'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping from title to index
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    title = data.get("title")

    if title not in indices:
        return jsonify({"error": "Assessment not found"}), 404

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    assessment_indices = [i[0] for i in sim_scores]

    recommendations = df.iloc[assessment_indices][['Title', 'Description']].to_dict(orient='records')
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
