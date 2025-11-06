import json, re, asyncio, numpy as np
import torch
from pdfminer.high_level import extract_text
from docx import Document
from sklearn.metrics import (
    r2_score, f1_score, precision_score, recall_score,
    accuracy_score, mean_absolute_error, mean_squared_error
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from crewai import Agent, Task, Crew

# -----------------------------
# HUGGING FACE MODEL CONFIG
# -----------------------------
BASE_MODEL_ID = "google/gemma-2b-it"
ADAPTER_MODEL_ID = "kirankumarpetlu/Fine_Tunned_LLM" 

# --- Create 4-bit quantization config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
# ----------------------------------------

print(f"Loading base model: {BASE_MODEL_ID} (with 4-bit quantization)...")
# Load the base model with quantization
# NOTE: 'token' argument removed. Use 'huggingface-cli login' in your terminal.
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,  # Loads in 4-bit
    device_map="auto"                # Manages device memory
)

print(f"Loading adapters: {ADAPTER_MODEL_ID}...")
# Load the PEFT model (adapters) on top
try:
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)
except OSError as e:
    print("\n" + "="*50)
    print(f"[ERROR] Could not load adapters from {ADAPTER_MODEL_ID}.")
    print("Please ensure this repository contains 'adapter_model.bin' and 'adapter_config.json'.")
    print(f"Details: {e}")
    print("="*50 + "\n")
    raise

print("Merging adapters into model for inference...")
# Merge the adapters into the base model for faster inference
model = model.merge_and_unload()

# Load the tokenizer from the base model
# NOTE: 'token' argument removed.
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True
)

print("Creating text-generation pipeline...")
# Create the pipeline with the *merged* model and the base tokenizer
pipe = pipeline(
    "text-generation", 
    model=model,  # Use the merged model object
    tokenizer=tokenizer, 
    trust_remote_code=True,
    device_map="auto"
)
print("Pipeline ready.")

# -----------------------------

def hf_call(prompt: str) -> str:
    """Generate output using your fine-tuned Hugging Face model."""
    try:
        # Gemma uses a specific chat template format.
        messages = [
            {"role": "user", "content": prompt},
        ]
        # Apply the chat template
        prompt_template = pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = pipe(
            prompt_template,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )
        
        # The output includes the prompt, so we need to slice it off
        full_text = outputs[0]["generated_text"]
        response = full_text[len(prompt_template):].strip()
        return response
    
    except Exception as e:
        return f"[ERROR] {e}"

def extract_json(text):
    """Extract JSON safely from model output."""
    if not text:
        return {}
    # Find the first '{' and the last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback for nested or malformed
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
    except Exception as e:
        print(f"Error parsing file {name}: {e}")
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
# TASK PIPELINE
# -----------------------------
# 1. Added 'resume_name' argument to accept the filename
def analyze_resume_with_crew(jd_text, resume_text, resume_name):
    """Run full CrewAI pipeline for a single resume."""
    
    # Define expected JSON formats to guide the LLM
    parser_format = '{"skills": ["skill1", ...], "experience": [{"role": "...", "company": "...", ...}], "education": [{"degree": "...", "institution": "...", ...}]}'
    skill_format = '{"technical_skills": ["skill", ...], "soft_skills": ["skill", ...]}'
    prof_format = '{"skill_justifications": [{"skill": "Python", "justification": "Used in project X..."}, ...]}'
    matcher_format = '{"matched_skills": [...], "missing_skills": [...], "fit_summary": "..."}'
    ranker_format = '{"overall_score": 85, "summary": "Candidate is a strong fit because..."}'

    try:
        # Truncate inputs to avoid excessive token usage
        jd_safe = jd_text[:3000]
        resume_safe = resume_text[:3000]

        parser_task = Task(
            description=f"Parse this resume into structured JSON.\nResume:\n{resume_safe}",
            expected_output=f"A single JSON object. Example: {parser_format}",
            agent=parser_agent,
        )

        skill_task = Task(
            description=f"Extract all skills and proficiencies. Format as JSON.\nResume:\n{resume_safe}",
            expected_output=skill_format,
            agent=skill_agent,
            context=[parser_task], 
        )

        proficiency_task = Task(
            description=f"Justify each skill's proficiency based on projects. Format as JSON.\nResume:\n{resume_safe}",
            expected_output=prof_format,
            agent=proficiency_agent,
            context=[skill_task],
        )

        matcher_task = Task(
            description=f"Match resume skills against job description. Format as JSON.\nJob Description:\n{jd_safe}\nResume:\n{resume_safe}",
            expected_output=matcher_format,
            agent=matcher_agent,
            context=[proficiency_task],
        )

        ranker_task = Task(
            description=f"Generate overall score (0-100) and reasoning summary. Format as JSON.\nMatching Results:\n{matcher_task.output}",
            expected_output=ranker_format,
            agent=ranker_agent,
            context=[matcher_task],
        )

        # 2. --- Define the crew with tasks ---
        # This is the modern way to run crewai
        local_crew = Crew(
            agents=[parser_agent, skill_agent, proficiency_agent, matcher_agent, ranker_agent],
            tasks=[parser_task, skill_task, proficiency_task, matcher_task, ranker_task]
        )

        # 3. --- Use .kickoff() instead of .run() ---
        # This fixes the "'Crew' object has no attribute 'run'" error
        result_output = local_crew.kickoff()
        
        # Combine all JSON outputs into one result
        final_json = {}
        final_json.update(extract_json(parser_task.output))
        final_json.update(extract_json(skill_task.output))
        final_json.update(extract_json(proficiency_task.output))
        final_json.update(extract_json(matcher_task.output))
        final_json.update(extract_json(result_output)) 
        
        # 4. --- Add the resume_name to the final dictionary ---
        # This fixes the "KeyError: 'resume_name'"
        final_json['resume_name'] = resume_name
        
        return final_json
        
    except Exception as e:
        print(f"Error during crew run: {e}")
        # 5. --- Also add resume_name to the error dictionary ---
        return {"error": str(e), "overall_score": 0, "resume_name": resume_name}


# -----------------------------
# ASYNC MULTI-RESUME HANDLER
# -----------------------------
async def rank_resumes(jd_text, upload_files):
    tasks = []
    for f in upload_files:
        resume_text = parse_resume_file(f)
        
        # 6. --- Get the resume's filename ---
        name = getattr(f, "name", "unknown_resume.txt")
        
        if not resume_text: # Skip if parsing failed
            print(f"Skipping file {name} due to parsing error or empty content.")
            continue
            
        # 7. --- Pass the 'name' into the task ---
        # This fixes the "missing 1 required positional argument: 'resume_name'" error
        tasks.append(asyncio.to_thread(analyze_resume_with_crew, jd_text, resume_text, name))
    
    results = await asyncio.gather(*tasks)
    metrics = compute_metrics(results, jd_text)
    return {"results": results, "metrics": metrics}


# -----------------------------
# METRICS CALCULATION
# -----------------------------
def compute_metrics(results, jd_text=""):
    if not results:
        return {}
    
    valid_results = [r for r in results if r and "overall_score" in r]
    if not valid_results:
        return {"error": "No valid results with 'overall_score' found."}

    # Use a placeholder "ground truth" for demonstration
    y_pred = np.array([r.get("overall_score", 0) for r in valid_results])
    
    # Avoid R^2 error if there's only one resume
    if len(y_pred) < 2:
        y_true = np.array([100]) # Dummy value, R2 will be undefined
        y_pred = np.array([y_pred[0]])
    else:
        y_true = np.linspace(100, 50, len(y_pred)) 

    metrics = {
        "R2": 0.0, # Default
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
    }
    if len(y_pred) >= 2:
         metrics["R2"] = round(r2_score(y_true, y_pred), 3)


    y_true_bin = (y_true >= 70).astype(int)
    y_pred_bin = (y_pred >= 70).astype(int)
    metrics.update({
        "Accuracy": round(accuracy_score(y_true_bin, y_pred_bin), 3),
        "F1": round(f1_score(y_true_bin, y_pred_bin, zero_division=0), 3),
        "Precision": round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 3),
        "Recall": round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 3),
    })

    matched = [" ".join(r.get("matched_skills", [])) for r in valid_results if r.get("matched_skills")]
    missing = [" ".join(r.get("missing_skills", [])) for r in valid_results if r.get("missing_skills")]
    
    if not matched and not missing:
        metrics["TFIDF_Skill_Similarity"] = 0.0
    else:
        try:
            vec_texts = matched + missing
            if jd_text:
                vec_texts.append(jd_text)
            
            vec = TfidfVectorizer().fit(vec_texts)
            
            if not matched or not jd_text:
                 metrics["TFIDF_Skill_Similarity"] = 0.0
            else:
                skill_sim = cosine_similarity(vec.transform(matched), vec.transform([jd_text])).mean()
                metrics["TFIDF_Skill_Similarity"] = round(float(skill_sim), 3)
        except Exception:
            metrics["TFIDF_Skill_Similarity"] = 0.0

    summaries = [r.get("summary", "") for r in valid_results if r.get("summary")]
    if not summaries:
        summaries = [r.get("fit_summary", "") for r in valid_results if r.get("fit_summary")]
    
    if not summaries or not jd_text:
         metrics["Semantic_Job_Resume_Similarity"] = 0.0
    else:
        try:
            vec = TfidfVectorizer().fit([jd_text] + summaries)
            sims = cosine_similarity(vec.transform([jd_text]), vec.transform(summaries))
            metrics["Semantic_Job_Resume_Similarity"] = round(float(np.mean(sims)), 3)
        except Exception:
            metrics["Semantic_Job_Resume_Similarity"] = 0.0

    # Calculate MAE contribution, avoid division by zero
    mae_score = (1 - metrics["MAE"] / 100) if metrics["MAE"] < 100 else 0.0

    metrics["Overall_Evaluation_Score"] = round(
        (0.4 * mae_score)
        + (0.3 * metrics["TFIDF_Skill_Similarity"])
        + (0.3 * metrics["Semantic_Job_Resume_Similarity"]),
        3,
    )
    return metrics

# -----------------------------
# EXAMPLE RUN (for testing)
# -----------------------------
if __name__ == "__main__":
    # This block runs only when you execute the script directly
    # (e.g., python resume_agent.py)
    
    print("\n--- Running local test ---")
    
    # Mock file objects for testing
    class MockFile:
        def __init__(self, name, content):
            self.name = name
            self._content = content
        def read(self):
            return self._content.encode('utf-8')
        
        # Add a dummy __str__ for better printing
        def __str__(self):
            return f"MockFile(name='{self.name}')"
        
        # Add dummy __repr__
        def __repr__(self):
            return self.__str__()

    # 1. Define a Job Description
    test_jd = """
    We are hiring a Senior Python Developer.
    Must have 5+ years of experience with Python, Django, and PostgreSQL.
    Strong knowledge of REST APIs and Docker is required.
    Experience with AWS and React is a big plus.
    """
    
    # 2. Define a Mock Resume
    test_resume_content = """
    John Doe
    Software Engineer
    
    Experience:
    - Senior Developer at TechCorp (2018-Present)
      - Built scalable web applications using Python, Django, and PostgreSQL.
      - Developed REST APIs for mobile clients.
      - Deployed services using Docker and Kubernetes.
      
    - Junior Developer at Webly (2016-2018)
      - Worked with Python and Flask.
      
    Education:
    - B.S. in Computer Science, State University
    
    Skills:
    - Python, Django, Flask, PostgreSQL, MySQL
    - Docker, Kubernetes, AWS
    - JavaScript, React
    """
    
    mock_resume_file = MockFile("john_doe_resume.txt", test_resume_content)
    
    # 3. Run the async function
    async def main():
        results = await rank_resumes(test_jd, [mock_resume_file])
        print("\n--- TEST RESULTS ---")
        print(json.dumps(results, indent=2))
        
    asyncio.run(main())