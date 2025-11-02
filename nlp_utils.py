import re
import pdfplumber
import docx
import filetype
import spacy
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# ------------------ BASIC DATA ------------------
COMMON_SKILLS = [
    "python","java","c++","c","javascript","react","angular","node","django",
    "flask","sql","mongodb","postgresql","aws","azure","gcp","docker","kubernetes",
    "machine learning","deep learning","nlp","pandas","numpy","scikit-learn",
    "tensorflow","pytorch","git","html","css","rest","api","microservices"
]

EDUCATION_KEYWORDS = [
    "bachelor","b.sc","btech","b.e","bs","b.a",
    "master","m.sc","m.tech","m.e","ms","mba","phd","doctor"
]

# ------------------ FILE READING ------------------
def read_pdf(path):
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text.append(txt)
    except Exception:
        pass
    return "\n".join(text)

def read_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def read_txt(path):
    try:
        with open(path, 'r', encoding='utf8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def file_to_text(path):
    kind = None
    try:
        kind = filetype.guess(path)
    except Exception:
        kind = None

    ext = path.rsplit('.',1)[-1].lower()
    if ext == 'pdf' or (kind and kind.mime == 'application/pdf'):
        return read_pdf(path)
    if ext == 'docx' or (kind and 'word' in (kind.mime or '')):
        return read_docx(path)
    if ext == 'txt' or ext == 'text':
        return read_txt(path)
    return read_txt(path)

# ------------------ DATA EXTRACTION ------------------
def extract_contact(text):
    email = None
    phone = None
    em = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if em:
        email = em.group(0)
    ph = re.search(r'(\+?\d[\d\-\s]{7,}\d)', text)
    if ph:
        phone = ph.group(0)
    return {'email': email, 'phone': phone}

def extract_skills(text, top_n=40):
    t = text.lower()
    found = set()
    for skill in COMMON_SKILLS:
        if skill in t:
            found.add(skill)
        else:
            ratio = fuzz.partial_ratio(skill, t)
            if ratio > 85:
                found.add(skill)
    return sorted(list(found))

def extract_education(text):
    t = text.lower()
    eds = [key for key in EDUCATION_KEYWORDS if key in t]
    return list(sorted(set(eds)))

def extract_experience_years(text):
    yrs = re.findall(r'(\d+)\s+years?', text.lower())
    yrs = [int(x) for x in yrs]
    if yrs:
        return max(yrs)
    ranges = re.findall(r'(\b(19|20)\d{2})\s*[-to]+\s*(\b(19|20)\d{2})', text)
    years = []
    for r in ranges:
        try:
            y1 = int(r[0])
            y2 = int(r[2])
            if y2 >= y1:
                years.append(y2 - y1)
        except:
            pass
    return max(years) if years else 0

def parse_resume_text(path):
    text = file_to_text(path)
    doc = nlp(text[:20000])
    contact = extract_contact(text)
    skills = extract_skills(text)
    education = extract_education(text)
    exp_years = extract_experience_years(text)
    return {
        "text_snippet": text[:2000],
        "contact": contact,
        "skills": skills,
        "education": education,
        "experience_years": exp_years,
    }

def extract_skills_from_jd(jd_text):
    t = jd_text.lower()
    found = [s for s in COMMON_SKILLS if s in t]
    return sorted(list(set(found)))

# ------------------ KNN MODEL ------------------
def get_vector(text):
    """Convert text into spaCy vector."""
    return nlp(text).vector

def train_knn():
    """Train demo KNN model on small text dataset."""
    resumes = [
        "MERN developer skilled in React, Node.js, and MongoDB",
        "Backend developer experienced in Flask and Python APIs",
        "Frontend React developer with Redux",
        "Data analyst familiar with SQL and PowerBI"
    ]
    jds = [
        "Looking for MERN full-stack developer skilled in React and Node",
        "Hiring data analyst with Excel and SQL experience",
        "Need backend developer for Python REST APIs",
        "React frontend developer required"
    ]
    labels = [1, 0, 1, 0]
    X = np.array([get_vector(r + " " + j) for r, j in zip(resumes, jds)])
    y = np.array(labels)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

knn_model = train_knn()

# ------------------ SCORING ------------------
def score_resume(resume_data, jd_skills):
    weights = {"skill": 0.5, "experience": 0.2, "education": 0.1, "knn": 0.2}

    resume_skills = set(resume_data.get("skills", []))
    jd_skills = set(jd_skills)
    resume_text = " ".join(resume_skills)
    jd_text = " ".join(jd_skills)

    # --- Skill Score ---
    if not jd_skills:
        skill_score = min(1.0, len(resume_skills)/5)
    else:
        matched = resume_skills.intersection(jd_skills)
        skill_score = len(matched)/max(1, len(jd_skills))

    # --- Experience Score ---
    exp = resume_data.get("experience_years", 0)
    if exp >= 5:
        exp_score = 1.0
    elif exp >= 2:
        exp_score = 0.7
    elif exp >= 1:
        exp_score = 0.4
    else:
        exp_score = 0.0

    # --- Education Score ---
    edu = resume_data.get("education", [])
    edu_score = 0.0
    if any(k in ["master","m.sc","m.tech","ms","mba","phd","doctor"] for k in edu):
        edu_score = 1.0
    elif any(k in ["bachelor","b.sc","btech","b.e","bs","b.a"] for k in edu):
        edu_score = 0.7

    # --- KNN Prediction ---
    combined_text = resume_text + " " + jd_text
    vector = get_vector(combined_text).reshape(1, -1)
    knn_pred = knn_model.predict(vector)[0]
    knn_score = 1.0 if knn_pred == 1 else 0.0

    # --- Final Weighted Score ---
    total = (
        skill_score * weights["skill"]
        + exp_score * weights["experience"]
        + edu_score * weights["education"]
        + knn_score * weights["knn"]
    )
    percent = round(total * 100, 1)

    return {
        "match_percent": percent,
        "skill_score": round(skill_score * 100, 1),
        "experience_score": round(exp_score * 100, 1),
        "education_score": round(edu_score * 100, 1),
        "knn_score": round(knn_score * 100, 1),
        "matched_skills": sorted(list(resume_skills.intersection(jd_skills))),
        "missing_skills": sorted(list(jd_skills - resume_skills)),
    }
