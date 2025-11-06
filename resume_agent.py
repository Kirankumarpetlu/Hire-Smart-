import json, re, asyncio, numpy as np
from pdfminer.high_level import extract_text
from docx import Document
from sklearn.metrics import (
    r2_score, f1_score, precision_score, recall_score,
    accuracy_score, mean_absolute_error, mean_squared_error
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, pipeline
from crewai import Agent, Task, Crew

# -----------------------------
# HUGGING FACE MODEL CONFIG
# -----------------------------
# MODEL_ID = "kirankumarpetlu/Fine_Tunned_LLM"
MODEL_ID="google/gemma-2b-it"
 

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
pipe = pipeline("text-generation", model=MODEL_ID, token=token,tokenizer=tokenizer, trust_remote_code=True)

def hf_call(prompt: str) -> str:
    """Generate output using your fine-tuned Hugging Face model."""
    try:
        outputs = pipe(
            prompt,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )
        return outputs[0]["generated_text"].strip()
    except Exception as e:
        return f"[ERROR] {e}"

def extract_json(text):
    """Extract JSON safely from model output."""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}

# -----------------------------
# FILE PARSING HELPERS
# -----------------------------
def parse_resume_file(upload_file):
    name = getattr(upload_file, "name", getattr(upload_file, "filename", "resume"))
    try:
        if name.lower().endswith(".pdf"):
            return extract_text(upload_file)
        elif name.lower().endswith(".docx"):
            doc = Document(upload_file)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            raw = upload_file.read()
            return raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    except Exception:
        return ""

# -----------------------------
# CREWAI AGENTS
# -----------------------------
parser_agent = Agent(
    role="Resume Parser",
    goal="Extract structured resume data such as skills, education, and experience.",
    backstory="Specialist in extracting structured information from resumes.",
    llm=hf_call,
)

skill_agent = Agent(
    role="Skill Extractor",
    goal="Identify technical and soft skills along with proficiency levels.",
    backstory="Understands domains and can assess skill categories.",
    llm=hf_call,
)

proficiency_agent = Agent(
    role="Proficiency Justifier",
    goal="Provide justifications for each skill's proficiency using project or experience details.",
    backstory="Analyzes work experience to justify skill levels with reasoning.",
    llm=hf_call,
)

matcher_agent = Agent(
    role="Job Matcher",
    goal="Compare resume skills with job description and identify matches, gaps, and overall fit.",
    backstory="Experienced recruiter comparing candidates with job roles.",
    llm=hf_call,
)

ranker_agent = Agent(
    role="Ranker Agent",
    goal="Compute overall candidate score and justification summary.",
    backstory="Senior analyst evaluating skill depth and job relevance.",
    llm=hf_call,
)

# -----------------------------
# CREW SETUP
# -----------------------------
crew = Crew(
    agents=[parser_agent, skill_agent, proficiency_agent, matcher_agent, ranker_agent]
)

# -----------------------------
# TASK PIPELINE
# -----------------------------
def analyze_resume_with_crew(jd_text, resume_text):
    """Run full CrewAI pipeline for a single resume."""
    parser_task = Task(
        description=f"Parse resume into structured JSON.\nResume:\n{resume_text[:3000]}",
        agent=parser_agent,
    )

    skill_task = Task(
        description=f"Extract all skills and proficiencies.\nResume:\n{resume_text[:3000]}",
        agent=skill_agent,
        depends_on=[parser_task],
    )

    proficiency_task = Task(
        description=f"Justify each skill's proficiency based on projects.\nResume:\n{resume_text[:3000]}\nSkills:\n{skill_task.output}",
        agent=proficiency_agent,
        depends_on=[skill_task],
    )

    matcher_task = Task(
        description=f"Match skills against job description.\nJob Description:\n{jd_text[:3000]}\nResume:\n{resume_text[:3000]}",
        agent=matcher_agent,
        depends_on=[proficiency_task],
    )

    ranker_task = Task(
        description=f"Generate overall score and reasoning summary.\nMatching Results:\n{matcher_task.output}",
        agent=ranker_agent,
        depends_on=[matcher_task],
    )

    result = crew.run([parser_task, skill_task, proficiency_task, matcher_task, ranker_task])
    return result

# -----------------------------
# METRICS CALCULATION
# -----------------------------
def compute_metrics(results, jd_text=""):
    if not results:
        return {}

    y_pred = np.array([r.get("overall_score", 0) for r in results])
    y_true = np.linspace(100, 50, len(y_pred))

    metrics = {
        "R2": round(r2_score(y_true, y_pred), 3),
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
    }

    y_true_bin = (y_true >= 70).astype(int)
    y_pred_bin = (y_pred >= 70).astype(int)
    metrics.update({
        "Accuracy": round(accuracy_score(y_true_bin, y_pred_bin), 3),
        "F1": round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 3),
        "Precision": round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 3),
        "Recall": round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 3),
    })

    # Skill similarity
    matched = [" ".join(r.get("matched_skills", [])) for r in results]
    missing = [" ".join(r.get("missing_skills", [])) for r in results]
    try:
        vec = TfidfVectorizer().fit(matched + missing + [jd_text])
        skill_sim = cosine_similarity(vec.transform(matched), vec.transform([jd_text])).mean()
        metrics["TFIDF_Skill_Similarity"] = round(float(skill_sim), 3)
    except Exception:
        metrics["TFIDF_Skill_Similarity"] = 0.0

    # Semantic similarity
    summaries = [r.get("summary", "") for r in results if r.get("summary")]
    try:
        vec = TfidfVectorizer().fit([jd_text] + summaries)
        sims = cosine_similarity(vec.transform([jd_text]), vec.transform(summaries))
        metrics["Semantic_Job_Resume_Similarity"] = round(float(np.mean(sims)), 3)
    except Exception:
        metrics["Semantic_Job_Resume_Similarity"] = 0.0

    metrics["Overall_Evaluation_Score"] = round(
        0.4 * (1 - metrics["MAE"] / 100)
        + 0.3 * metrics["TFIDF_Skill_Similarity"]
        + 0.3 * metrics["Semantic_Job_Resume_Similarity"],
        3,
    )
    return metrics

# -----------------------------
# ASYNC MULTI-RESUME HANDLER
# -----------------------------
async def rank_resumes(jd_text, upload_files):
    tasks = []
    for f in upload_files:
        resume_text = parse_resume_file(f)
        tasks.append(asyncio.to_thread(analyze_resume_with_crew, jd_text, resume_text))
    results = await asyncio.gather(*tasks)
    metrics = compute_metrics(results, jd_text)
    return {"results": results, "metrics": metrics}
