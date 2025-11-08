import os
import re
import time
import torch
import tempfile
import pdfplumber
import docx
import asyncio
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "kirankumarpetlu/Fine_Tunned_LLM"
MAX_PROMPT_TOKENS = 1500
GEN_TOKENS_ANALYZE = 300
GEN_TOKENS_PROF = 350
GEN_TOKENS_COMPARE = 180
GEN_TOKENS_RANK = 120

# ============================================================
# INIT MODELS
# ============================================================
print(f"🔹 [INIT] Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
torch.set_num_threads(4)
print(f"✅ [INIT] Model loaded on: {device.upper()}")

# ============================================================
# TEXT GENERATION
# ============================================================
def hf_generate(prompt: str, max_new_tokens: int = 128) -> str:
    """Generate text from model and strip echoed prompt."""
    print("⚡ [GEN] Generating text...")
    start = time.time()
    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_TOKENS).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.4,
            top_p=0.9,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in text:
        text = text.replace(prompt, "")
    if "{" in text:
        text = text[text.find("{"):]
    print(f"✅ [GEN DONE] Took {round(time.time()-start, 2)}s, Output length: {len(text)} chars\n")
    return text.strip()

# ============================================================
# SAFE JSON PARSER + FALLBACK
# ============================================================
def safe_json_parse(output: str):
    """Try to parse JSON robustly from text."""
    if not output:
        return None
    try:
        return json.loads(output)
    except Exception:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None

def heuristic_parse_plain_text(text: str):
    """Fallback parser when no JSON is found."""
    result = {
        "skills": [],
        "technical_skills": [],
        "soft_skills": [],
        "experience": "",
        "education": "",
        "projects": [],
        "achievements": [],
        "certifications": [],
    }

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    current = None
    sections = {k: [] for k in result.keys()}

    for l in lines:
        low = l.lower()
        if any(x in low for x in ["skills", "expertise", "technologies"]):
            current = "skills"; continue
        elif "experience" in low:
            current = "experience"; continue
        elif "education" in low:
            current = "education"; continue
        elif "project" in low:
            current = "projects"; continue
        elif any(x in low for x in ["achievement", "award"]):
            current = "achievements"; continue
        elif "certification" in low:
            current = "certifications"; continue
        if current:
            sections[current].append(l)

    if sections["skills"]:
        parts = re.split(r"[,;|•\-]", " ".join(sections["skills"]))
        result["skills"] = [p.strip() for p in parts if len(p.strip()) > 1]
    result["experience"] = " ".join(sections["experience"])[:1500]
    result["education"] = " ".join(sections["education"])[:800]
    result["projects"] = sections["projects"][:10]
    result["achievements"] = sections["achievements"][:10]
    result["certifications"] = sections["certifications"][:10]
    return result

# ============================================================
# FILE TEXT EXTRACTION
# ============================================================
def extract_text(file):
    print(f"📂 [EXTRACT] Reading file: {file.name}")
    start = time.time()
    name = file.name.lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    text = ""
    if name.endswith(".pdf"):
        with pdfplumber.open(tmp_path) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
    elif name.endswith(".docx"):
        doc = docx.Document(tmp_path)
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        text = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()

    os.remove(tmp_path)
    print(f"✅ [EXTRACT DONE] {len(text)} chars in {round(time.time()-start,2)}s\n")
    return text.strip()

# ============================================================
# ⚙️ SEMANTIC SIMILARITY FUNCTION
# ============================================================
def get_semantic_score(jd_text: str, resume_text: str) -> float:
    """Compute cosine similarity (0–100) between JD and Resume."""
    try:
        start = time.time()
        jd_emb = embedder.encode(jd_text, convert_to_tensor=True)
        res_emb = embedder.encode(resume_text, convert_to_tensor=True)
        score = float(util.pytorch_cos_sim(jd_emb, res_emb).item() * 100)
        print(f"📊 [SIMILARITY] Score: {round(score,2)} (computed in {round(time.time()-start,2)}s)")
        return round(score, 2)
    except Exception as e:
        print(f"⚠️ [SIM ERROR] {e}")
        return 0.0

# ============================================================
# 🧠 RESUME ANALYSIS
# ============================================================
def analyze_resume(resume_text: str) -> dict:
    print("🧩 [STEP] Analyzing resume -> structured JSON")
    example = {
        "skills": ["Python", "Django"],
        "technical_skills": ["Python", "Django"],
        "soft_skills": ["Teamwork", "Communication"],
        "experience": "2 years backend developer",
        "education": "B.Tech in Computer Science",
        "projects": ["AI chatbot", "Fraud Detection"],
        "achievements": ["Published paper"],
        "certifications": ["AWS Developer"]
    }
    prompt = (
        "You are an expert resume parser. Return ONLY valid JSON (no explanation) "
        f"matching this structure:\n{json.dumps(example, indent=2)}\n\n"
        f"Resume:\n{resume_text}"
    )
    raw = hf_generate(prompt, max_new_tokens=GEN_TOKENS_ANALYZE)
    parsed = safe_json_parse(raw)
    if parsed is None:
        print("⚠️ [ERROR] JSON parsing failed. Using heuristic fallback.")
        parsed = heuristic_parse_plain_text(resume_text)
    return parsed

# ============================================================
# 🧩 SKILL PROFICIENCY (Reasoning-Based)
# ============================================================
def generate_skill_proficiency(resume_data: dict, resume_text: str) -> dict:
    """
    Generate detailed reasoning-based proficiency explanations for each skill.
    """
    print("🧩 [STEP] Generating skill proficiency details...")
    skills = resume_data.get("skills", [])
    if not skills:
        print("⚠️ [WARN] No skills found — skipping proficiency.")
        return {"skills_proficiency_details": []}

    skill_str = ", ".join(skills[:12])
    prompt = f"""
You are an expert HR analyst. For each skill, analyze the resume text and write a short justification
(1–3 sentences) describing how the candidate demonstrated that skill in their projects, work, or education.
Be specific and realistic — refer to the context where possible.
Return ONLY valid JSON in this format:
{{
  "skills_proficiency_details": [
    {{
      "name": "Python",
      "proficiency_detail": "Built machine learning models and APIs using Python and Flask, showing strong backend experience."
    }},
    {{
      "name": "Docker",
      "proficiency_detail": "Used Docker for containerizing web applications, demonstrating DevOps knowledge."
    }}
  ]
}}

SKILLS: {skill_str}

RESUME TEXT:
{resume_text[:2000]}
"""
    raw = hf_generate(prompt, max_new_tokens=GEN_TOKENS_PROF)
    parsed = safe_json_parse(raw)

    # ✅ Context-aware fallback
    if parsed is None or "skills_proficiency_details" not in parsed:
        details = []
        lower_resume = resume_text.lower()
        for s in skills[:12]:
            match = re.search(rf"([^.]*\b{s.split()[0].lower()}\b[^.]*)\.", lower_resume)
            if match:
                context = match.group(1).strip().capitalize()
                detail = f"Demonstrated {s} in resume context: \"{context}.\""
            else:
                detail = f"Has working knowledge of {s}, as mentioned in projects or experience."
            details.append({"name": s, "proficiency_detail": detail})
        parsed = {"skills_proficiency_details": details}

    print(f"✅ [PROFICIENCY] Generated {len(parsed.get('skills_proficiency_details', []))} reasoning entries.")
    return parsed

# ============================================================
# 🧩 JD COMPARISON
# ============================================================
def compare_with_jd(resume_data: dict, jd_text: str) -> dict:
    print("🧩 [STEP] Comparing resume with JD...")
    skills = resume_data.get("skills", [])
    summary = f"Skills: {', '.join(skills)}\nExperience: {resume_data.get('experience', '')}"
    example = {
        "matched_skills": ["Python"],
        "missing_skills": ["Kubernetes"],
        "fit_summary": "Good backend match, missing DevOps."
    }
    prompt = (
        "Compare JOB DESCRIPTION and RESUME SUMMARY. Return ONLY valid JSON like:\n"
        f"{json.dumps(example, indent=2)}\n\nJOB:\n{jd_text}\n\nRESUME:\n{summary}"
    )
    raw = hf_generate(prompt, max_new_tokens=GEN_TOKENS_COMPARE)
    parsed = safe_json_parse(raw)
    if parsed is None:
        jd_lower = jd_text.lower()
        matched = [s for s in skills if s.lower() in jd_lower]
        missing = [s for s in skills if s.lower() not in jd_lower]
        parsed = {"matched_skills": matched, "missing_skills": missing, "fit_summary": f"Matched {len(matched)} skills."}
    print(f"✅ [COMPARE] matched: {len(parsed.get('matched_skills', []))}")
    return parsed

# ============================================================
# 🧩 CANDIDATE RANKING
# ============================================================
def rank_candidate(jd_text: str, resume_text: str, comparison: dict) -> dict:
    print("🧩 [STEP] Ranking candidate...")
    base_score = get_semantic_score(jd_text, resume_text)
    example = {"overall_score": 88, "summary": "Good fit", "justification": "Strong skills match."}
    prompt = (
        "Based on JD and comparison, return ONLY valid JSON:\n"
        f"{json.dumps(example, indent=2)}\n\nJOB:\n{jd_text}\n\nCOMPARE:\n{json.dumps(comparison)}"
    )
    raw = hf_generate(prompt, max_new_tokens=GEN_TOKENS_RANK)
    parsed = safe_json_parse(raw)
    if parsed is None:
        parsed = {"overall_score": base_score, "summary": f"Auto summary (score={base_score})", "justification": "Auto-generated."}
    parsed["overall_score"] = parsed.get("overall_score", base_score)
    print(f"✅ [RANK] Final Score: {parsed['overall_score']}")
    return parsed

# ============================================================
# 🧩 PROCESS SINGLE RESUME
# ============================================================
async def process_resume(jd_text, resume_file):
    name = resume_file.name
    print(f"\n📄 [PROCESS] Starting: {name}")
    start = time.time()

    resume_text = extract_text(resume_file)
    resume_data = analyze_resume(resume_text)
    proficiency = generate_skill_proficiency(resume_data, resume_text)
    comparison = compare_with_jd(resume_data, jd_text)
    result = rank_candidate(jd_text, resume_text, comparison)

    elapsed = round(time.time() - start, 2)
    print(f"✅ [PROCESS DONE] {name} analyzed in {elapsed}s\n")

    return {
        "resume_name": name,
        "skills": resume_data.get("skills", []),
        "experience": resume_data.get("experience", ""),
        "education": resume_data.get("education", ""),
        "projects": resume_data.get("projects", []),
        "achievements": resume_data.get("achievements", []),
        "certifications": resume_data.get("certifications", []),
        "skills_proficiency_details": proficiency.get("skills_proficiency_details", []),
        "matched_skills": comparison.get("matched_skills", []),
        "missing_skills": comparison.get("missing_skills", []),
        "fit_summary": comparison.get("fit_summary", ""),
        "overall_score": result.get("overall_score", 0),
        "summary": result.get("summary", ""),
        "justification": result.get("justification", ""),
    }

# ============================================================
# ⚡ BATCH RANKING (for Streamlit)
# ============================================================
async def rank_resumes(jd_text, resume_files):
    print("\n🚀 [SYSTEM] Starting Resume Ranking Process...")
    results = []
    for i, f in enumerate(resume_files, start=1):
        print(f"==============================\n⚙️ [{i}/{len(resume_files)}] {f.name}\n==============================")
        res = await process_resume(jd_text, f)
        results.append(res)
        if device == "cuda":
            torch.cuda.empty_cache()
    print("✅ [SYSTEM] All resumes processed.\n")
    return results

# ============================================================
# CLI TEST
# ============================================================
if __name__ == "__main__":
    print("✅ Backend ready (Full Parsing + Reasoned Skill Proficiency + Ranking)")
