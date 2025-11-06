import os
import re
import torch
import tempfile
import pdfplumber
import docx
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import time

# ============================================================
# 🚀 1. Load Fine-Tuned Hugging Face Model (Optimized for CPU)
# ============================================================
MODEL_NAME = "kirankumarpetlu/Fine_Tunned_LLM"

print(f"🔹 [INIT] Loading fine-tuned model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # ✅ Important: disable dropout for inference
print(f"✅ [INIT] Model loaded successfully on: {device.upper()}")

# ============================================================
# 🧠 2. Safe Text Generation with Debugging + CPU Optimizations
# ============================================================
def hf_generate(prompt: str, max_new_tokens: int = 128) -> str:
    """Generate text using your fine-tuned Gemma model safely and fast."""
    print("\n⚡ [GENERATION] Generating text from model...")
    start_time = time.time()

    # Use inference mode for speed
    with torch.inference_mode():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elapsed = round(time.time() - start_time, 2)
    print(f"✅ [GENERATION DONE] Took {elapsed}s, Output length: {len(generated_text)} chars\n")
    return generated_text

# ============================================================
# 📄 3. File Text Extraction (PDF / DOCX / TXT)
# ============================================================
def extract_text(file):
    """Extract text content from resumes and job descriptions."""
    print(f"📂 [EXTRACT] Reading file: {file.name}")
    start_time = time.time()
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
        text = "\n".join([p.text for p in doc.paragraphs])
    elif name.endswith(".txt"):
        text = open(tmp_path, "r", encoding="utf-8").read()

    os.remove(tmp_path)
    print(f"✅ [EXTRACT DONE] {len(text)} characters extracted in {round(time.time() - start_time, 2)}s\n")
    return text.strip()

# ============================================================
# ⚙️ 4. Helper: Semantic Similarity (for JD vs Resume)
# ============================================================
def get_semantic_score(jd_text, resume_text):
    """Compute cosine similarity score between JD and Resume embeddings."""
    start_time = time.time()
    jd_emb = embedder.encode(jd_text, convert_to_tensor=True)
    res_emb = embedder.encode(resume_text, convert_to_tensor=True)
    score = round(util.pytorch_cos_sim(jd_emb, res_emb).item() * 100, 2)
    print(f"📊 [SIMILARITY] Score: {score} (computed in {round(time.time() - start_time, 2)}s)")
    return score

# ============================================================
# 🤖 5. Resume Analysis Functions (Pure Hugging Face)
# ============================================================
def analyze_resume(resume_text: str) -> dict:
    """Extract structured data (skills, education, etc.) from resume."""
    print("🧩 [STEP 1] Analyzing resume content...")
    prompt = f"""
    Analyze the following resume and return structured JSON with:
    - skills (technical + soft)
    - experience summary
    - education
    - key achievements
    Resume Text:
    {resume_text[:2000]}  # limit for safety
    """
    output = hf_generate(prompt, max_new_tokens=128)
    try:
        data = eval(output) if isinstance(output, str) else {}
    except Exception:
        print("⚠️ [ERROR] Failed to parse structured JSON from analysis output.")
        data = {"skills": [], "experience": "", "education": "", "achievements": ""}
    return data

def compare_with_jd(resume_data: dict, jd_text: str) -> dict:
    """Compare extracted resume data with job description."""
    print("🧩 [STEP 2] Comparing resume with JD...")
    resume_text = f"Skills: {resume_data.get('skills')}\nExperience: {resume_data.get('experience')}"
    prompt = f"""
    Compare the following resume and job description.
    Return JSON with:
    - matched_skills
    - missing_skills
    - fit_summary
    Job Description:
    {jd_text[:1500]}
    Resume:
    {resume_text[:1500]}
    """
    output = hf_generate(prompt, max_new_tokens=128)
    try:
        data = eval(output) if isinstance(output, str) else {}
    except Exception:
        print("⚠️ [ERROR] Failed to parse JD comparison JSON.")
        data = {"matched_skills": [], "missing_skills": [], "fit_summary": ""}
    return data

def rank_candidate(jd_text: str, resume_text: str, comparison: dict) -> dict:
    """Generate a final candidate score and justification."""
    print("🧩 [STEP 3] Ranking candidate...")
    base_score = get_semantic_score(jd_text, resume_text)
    prompt = f"""
    Based on this job description and resume comparison:
    JD: {jd_text[:1500]}
    Comparison: {comparison}
    Provide JSON with:
    - overall_score (0-100)
    - summary
    - justification
    """
    output = hf_generate(prompt, max_new_tokens=64)
    try:
        data = eval(output)
    except Exception:
        print("⚠️ [ERROR] Failed to parse ranking JSON, fallback to defaults.")
        data = {"overall_score": base_score, "summary": output, "justification": "Auto-generated."}
    data["overall_score"] = data.get("overall_score", base_score)
    print(f"✅ [RANKING DONE] Final Score: {data['overall_score']}")
    return data

# ============================================================
# 🧩 6. Process Each Resume
# ============================================================
async def process_resume(jd_text, resume_file):
    resume_name = resume_file.name
    print(f"\n📄 [PROCESS] Starting analysis for: {resume_name}")
    start_time = time.time()

    resume_text = extract_text(resume_file)
    resume_data = analyze_resume(resume_text)
    comparison = compare_with_jd(resume_data, jd_text)
    result = rank_candidate(jd_text, resume_text, comparison)

    elapsed = round(time.time() - start_time, 2)
    print(f"✅ [PROCESS DONE] {resume_name} analyzed in {elapsed}s\n")

    return {
        "resume_name": resume_name,
        "skills": resume_data.get("skills", []),
        "experience": resume_data.get("experience", ""),
        "education": resume_data.get("education", ""),
        "matched_skills": comparison.get("matched_skills", []),
        "missing_skills": comparison.get("missing_skills", []),
        "fit_summary": comparison.get("fit_summary", ""),
        "overall_score": result.get("overall_score", 0),
        "summary": result.get("summary", ""),
        "justification": result.get("justification", ""),
    }

# ============================================================
# ⚡ 7. Async Ranking for Multiple Resumes (Sequential CPU Safe)
# ============================================================
async def rank_resumes(jd_text, resume_files):
    """Process resumes sequentially (CPU-optimized + debug logs)."""
    print("\n🚀 [SYSTEM] Starting Resume Ranking Process...")
    results = []
    for i, file in enumerate(resume_files, start=1):
        print(f"==============================")
        print(f"⚙️ [{i}/{len(resume_files)}] Processing: {file.name}")
        print(f"==============================")
        result = await process_resume(jd_text, file)
        results.append(result)
        if device == "cuda":
            torch.cuda.empty_cache()  # free VRAM
    print("\n✅ [SYSTEM] All resumes processed successfully.\n")
    return results

# ============================================================
# 🧪 8. CLI Test
# ============================================================
if __name__ == "__main__":
    print("✅ Backend ready: Hugging Face only (Debug + CPU optimized).")
