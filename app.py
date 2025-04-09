from flask import Flask, request, render_template
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

df = pd.read_csv("assessments.csv")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []

    if request.method == 'POST':
        job_desc = request.form['job_desc']
        if job_desc.strip() != "":
            job_tfidf = vectorizer.transform([job_desc])
            similarity = cosine_similarity(job_tfidf, tfidf_matrix).flatten()

            top_indices = similarity.argsort()[-3:][::-1]
            for i in top_indices:
                score = similarity[i]
                if score < 0.2:
                    continue
                recommendations.append({
                    'name': df.iloc[i]['Assessment_Name'],
                    'description': df.iloc[i]['Description'],
                    'duration': df.iloc[i]['Duration'],
                    'adaptive': df.iloc[i]['Adaptive_Support'],
                    'remote': df.iloc[i]['Remote_Support'],
                    'test_type': df.iloc[i]['Test_Type'],
                    'url': df.iloc[i]['URL'],
                    'score': round(score, 2)
                })

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # use environment variable or fallback to 5000
    app.run(host='0.0.0.0', port=port)
