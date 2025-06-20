import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    return list(set([token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop]))

def calculate_ats_score(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    matches = set(resume_keywords).intersection(set(jd_keywords))
    score = (len(matches) / len(jd_keywords)) * 100 if jd_keywords else 0
    return round(score, 2)

def update_resume(resume_text, job_description):
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(job_description)
    missing_keywords = set(jd_keywords) - set(resume_keywords)
    
    enhancement_section = "\n\n--- Added Keywords for ATS Optimization ---\n"
    enhancement_section += ", ".join(missing_keywords)
    
    return resume_text + enhancement_section
