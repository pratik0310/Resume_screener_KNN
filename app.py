import os
import re
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from nlp_utils import parse_resume_text, extract_skills_from_jd, score_resume

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Job description or required skills
        jd_text = request.form.get('jobdesc', '').strip()
        file = request.files.get('resume')
        if not file or file.filename == '':
            return render_template('index.html', error="Please upload a resume file and paste job description.")
        if not allowed_file(file.filename):
            return render_template('index.html', error="Unsupported file type. Use PDF, DOCX or TXT.")
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # parse resume into text and extract info
        resume_data = parse_resume_text(path)

        # extract job skills
        jd_skills = extract_skills_from_jd(jd_text)

        # score resume
        result = score_resume(resume_data, jd_skills)

        return render_template('result.html',
                               resume_data=resume_data,
                               jd_text=jd_text,
                               jd_skills=jd_skills,
                               result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
