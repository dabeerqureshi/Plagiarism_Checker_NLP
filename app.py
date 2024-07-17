from flask import Flask, render_template, request
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def vectorize(text):
    return TfidfVectorizer().fit_transform(text).toarray()

def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file_post():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('file[]')
        student_files = []
        student_notes = []
        
        for file in uploaded_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            student_files.append(file.filename)
            with open(file_path, encoding='utf-8') as f:
                student_notes.append(f.read())
        
        vectors = vectorize(student_notes)
        s_vectors = list(zip(student_files, vectors))
        plagiarism_results = set()

        for student_a, text_vector_a in s_vectors:
            new_vectors = s_vectors.copy()
            current_index = new_vectors.index((student_a, text_vector_a))
            del new_vectors[current_index]
            for student_b, text_vector_b in new_vectors:
                sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                sim_percentage = sim_score * 100  # Convert to percentage
                student_pair = sorted((student_a, student_b))
                score = (student_pair[0], student_pair[1], sim_percentage)
                plagiarism_results.add(score)

        return render_template('result.html', results=plagiarism_results)

if __name__ == '__main__':
    app.run(debug=True)
