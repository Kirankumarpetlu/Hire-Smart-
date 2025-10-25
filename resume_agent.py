# resume_agent.py
import json
from pdfminer.high_level import extract_text
from docx import Document
from groq import Groq

# -----------------------------
# Groq client
# -----------------------------
GROQ_API_KEY = "paste your LLM's API "
groq_client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# Helper functions
# -----------------------------
def groq_call(prompt):
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def parse_resume_file(file):
    if file.name.endswith(".pdf"):
        return extract_text(file)
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return file.read().decode("utf-8", errors="ignore")

# -----------------------------
# Crew-style Agents with metadata
# -----------------------------
parser_agent = {
    "role": "Resume Parser",
    "goal": "Extract structured text (skills, experience, education) from resumes",
    "backstory": "Expert in cleaning and organizing resume data",
    "run": lambda resume_text: groq_call(f"Extract structured text (skills, experience, education) from this resume:\n{resume_text[:4000]}")
}

skill_agent = {
    "role": "Skill Extractor",
    "goal": "Extract skills and proficiency levels from resumes",
    "backstory": "Identifies technical, soft, and domain-specific skills",
    "run": lambda parsed_resume: groq_call(f"""
Extract all skills (technical, soft, domain-specific) from the resume below.
Provide estimated proficiency levels.
Return JSON ONLY in this format:
{{
  "skills": [
    {{"name": "Python", "proficiency": "Expert"}}
  ]
}}

Resume text:
{parsed_resume}
""")
}

proficiency_agent = {
    "role": "Proficiency Justifier",
    "goal": "Provide detailed justification of proficiency per skill based on projects/work experience",
    "backstory": "Analyzes work/projects in resume to quantify skill proficiency",
    "run": lambda parsed_resume, skills_json: groq_call(f"""
Given the resume text: {parsed_resume} and the skills JSON: {skills_json},
for each skill, provide detailed justification of proficiency based on work, projects, and experience.
Return JSON ONLY in this format:
{{
  "skills_proficiency_details": [
    {{"name": "Python", "proficiency_detail": "Worked on X, Y, Z projects"}}
  ]
}}
""")
}

matcher_agent = {
    "role": "JD Matcher",
    "goal": "Compare candidate skills with job description",
    "backstory": "Evaluates candidate fit based on matched and missing skills",
    "run": lambda skills_proficiency_json, jd_text: groq_call(f"""
Compare the following job description with candidate skills:
Job Description: {jd_text}
Candidate Skills JSON: {skills_proficiency_json}

Return JSON ONLY with:
{{
  "matched_skills": [...],
  "missing_skills": [...],
  "similarity_score": 0-100
}}
""")
}

ranker_agent = {
    "role": "Ranker Agent",
    "goal": "Produce overall candidate score and justification",
    "backstory": "Aggregates all previous agent outputs and provides final evaluation",
    "run": lambda matching_json, skills_proficiency_json: groq_call(f"""
Based on matching JSON: {matching_json} and skill proficiency: {skills_proficiency_json},
return final evaluation JSON ONLY:
{{
  "overall_score": 0-100,
  "proficiency_summary": "string",
  "justification": "string"
}}
""")
}

# -----------------------------
# Pipeline
# -----------------------------
def analyze_resume(jd_text, resume_text):
    parsed_resume = parser_agent["run"](resume_text)
    skills_json = skill_agent["run"](parsed_resume)
    skills_proficiency_json = proficiency_agent["run"](parsed_resume, skills_json)
    matching_json = matcher_agent["run"](skills_proficiency_json, jd_text)
    final_eval = ranker_agent["run"](matching_json, skills_proficiency_json)

    try:
        skills_json = json.loads(skills_json)
    except:
        skills_json = {"skills": []}
    try:
        skills_proficiency_json = json.loads(skills_proficiency_json)
    except:
        skills_proficiency_json = {"skills_proficiency_details": []}
    try:
        matching_json = json.loads(matching_json)
    except:
        matching_json = {"matched_skills": [], "missing_skills": [], "similarity_score": 0}
    try:
        final_eval = json.loads(final_eval)
    except:
        final_eval = {"overall_score": 0, "proficiency_summary": "", "justification": ""}

    return {
        "matched_skills": matching_json.get("matched_skills", []),
        "missing_skills": matching_json.get("missing_skills", []),
        "overall_score": final_eval.get("overall_score", 0),
        "proficiency_summary": final_eval.get("proficiency_summary", ""),
        "justification": final_eval.get("justification", ""),
        "skills_proficiency_details": skills_proficiency_json.get("skills_proficiency_details", [])
    }

def rank_resumes(job_description, resume_files):
    results = []
    for file in resume_files:
        resume_text = parse_resume_file(file)
        result = analyze_resume(job_description, resume_text)
        result["resume_name"] = file.name
        results.append(result)
    results.sort(key=lambda x: x["overall_score"], reverse=True)
    return results
