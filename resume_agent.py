# --- 1. THIS IS THE NEW ENVIRONMENT FIX ---
import os
# Use .pop() to completely remove the keys if they exist
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("OPENAI_API_BASE", None)
os.environ.pop("OPENAI_MODEL_NAME", None)
# ----------------------------------------

import json
import re
import asyncio
import numpy as np
from pdfminer.high_level import extract_text
from docx import Document
from transformers import AutoTokenizer, pipeline
from crewai import Agent, Task, Crew, Process  # Make sure Process is imported

# -----------------------------
# HUGGING FACE MODEL CONFIG
# -----------------------------
MODEL_ID = "kirankumarpetlu/Fine_Tunned_LLM"
# MODEL_ID="google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

pipe = pipeline(
    "text-generation", 
    model=MODEL_ID, 
    tokenizer=tokenizer, 
    trust_remote_code=True
)

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
        full_text = outputs[0]["generated_text"].strip()
        
        if full_text.startswith(prompt):
            return full_text[len(prompt):].strip()
        else:
            return full_text

    except Exception as e:
        print(f"[ERROR in hf_call] {e}")
        return f"[ERROR] {e}"

# -----------------------------
# JSON EXTRACTOR
# -----------------------------
def extract_json(text):
    """Extract JSON safely from model output."""
    if not text:
        return {}
    
    try:
        start_index = text.find('{')
        if start_index == -1:
            print("--- DEBUG: extract_json --- NO '{' found in text.")
            return {}
        
        end_index = text.rfind('}')
        if end_index == -1:
            print("--- DEBUG: extract_json --- NO '}' found in text.")
            return {}

        if end_index < start_index:
            print("--- DEBUG: extract_json --- '}' found before '{'. Invalid structure.")
            return {}

        json_str = text[start_index : end_index + 1]
        return json.loads(json_str)
    
    except json.JSONDecodeError as e:
        print(f"--- DEBUG: extract_json --- JSONDecodeError: {e}")
        print(f"--- FAILED ON STRING (first 200 chars): {json_str[:200]}...")
        return {}
    except Exception as e:
        print(f"--- DEBUG: extract_json --- An unexpected error occurred: {e}")
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
# --- 2. THIS IS THE NEW AGENT FIX ---
# -----------------------------
parser_agent = Agent(
    role="Resume Parser",
    goal="Extract structured resume data such as skills, education, and experience.",
    backstory="Specialist in extracting structured information from resumes. You only output valid JSON.",
    llm=hf_call,
    allow_delegation=False,
    verbose=True,
    fallback_llm=None  # <-- EXPLICITLY DISABLE FALLBACK
)

proficiency_agent = Agent(
    role="Proficiency Justifier",
    goal="Provide justifications for each skill's proficiency using project or experience details.",
    backstory="Analyzes work experience to justify skill levels with reasoning. You only output valid JSON.",
    llm=hf_call,
    allow_delegation=False,
    verbose=True,
    fallback_llm=None  # <-- EXPLICITLY DISABLE FALLBACK
)

matcher_agent = Agent(
    role="Job Matcher",
    goal="Compare resume skills with job description and identify matches, gaps, and overall fit.",
    backstory="Experienced recruiter comparing candidates with job roles. You only output valid JSON.",
    llm=hf_call,
    allow_delegation=False,
    verbose=True,
    fallback_llm=None  # <-- EXPLICITLY DISABLE FALLBACK
)

ranker_agent = Agent(
    role="Ranker Agent",
    goal="Compute overall candidate score and justification summary, and aggregate all data into a single JSON.",
    backstory="Senior analyst evaluating skill depth and job relevance, responsible for the final candidate report. You only output valid JSON.",
    llm=hf_call,
    allow_delegation=False,
    verbose=True,
    fallback_llm=None  # <-- EXPLICITLY DISABLE FALLBACK
)

# -----------------------------
# TASK PIPELINE
# -----------------------------
def analyze_resume_with_crew(jd_text, resume_text, resume_name):
    """Run full CrewAI pipeline for a single resume using .kickoff()"""
    
    print(f"\n--- 🚀 Starting Analysis for: {resume_name} ---")

    try:
        # --- Task 1: Parse Resume ---
        parser_task = Task(
            description=f"""
            Analyze the following resume and extract the information as a JSON object.
            ONLY output a single, valid JSON object. Do not add any conversational text.
            Resume:
            {resume_text[:3000]}
            """,
            agent=parser_agent,
            expected_output="A structured JSON with keys: 'name', 'summary', 'technical_skills', 'soft_skills', 'experience', 'education'."
        )

        # --- Task 2: Skill Proficiency ---
        proficiency_task = Task(
            description=f"""
            Based on the parsed resume context from the previous task, justify the 
            proficiency for each skill found.
            ONLY output a single, valid JSON object with a single key 'skills_proficiency_details'.
            Do not add any conversational text.
            """,
            agent=proficiency_agent,
            expected_output="A JSON object with a single key 'skills_proficiency_details'. Example: {'skills_proficiency_details': [{'skill': 'Python', 'justification': 'Used in project X...'}, ...]}",
            context=[parser_task] 
        )

        # --- Task 3: Job Matcher ---
        matcher_task = Task(
            description=f"""
            Using the parsed skills from the first task's context, match them 
            against the job description.
            ONLY output a single, valid JSON object with 'matched_skills' and 'missing_skills'.
            Job Description:
            {jd_text[:3000]}
            Resume:
            {resume_text[:3000]}
            """,
            agent=matcher_agent,
            expected_output="A JSON object with 'matched_skills' (list) and 'missing_skills' (list).",
            context=[parser_task] 
        )

        # --- Task 4: Final Ranker & Aggregator ---
        ranker_task = Task(
            description=f"""
            You are the final agent. You must aggregate all information from the previous tasks
            (parsed resume, skill justifications, and job matching results) into a
            single, comprehensive JSON object.
            
            Calculate an 'overall_score' (0-100), write a 'summary', and a 'justification'
            based on the matching results.
            
            ONLY output a single, final JSON object that includes *all* of the following keys:
            'technical_skills', 'soft_skills', 'experience', 'education',
            'skills_proficiency_details', 'matched_skills', 'missing_skills',
            'overall_score', 'summary', 'justification'.
            """,
            agent=ranker_agent,
            expected_output="A single, final JSON object containing all specified keys.",
            context=[parser_task, proficiency_task, matcher_task] 
        )

        # --- Create Crew and Kickoff ---
        crew = Crew(
            agents=[parser_agent, proficiency_agent, matcher_agent, ranker_agent],
            tasks=[parser_task, proficiency_task, matcher_task, ranker_task],
            verbose=True,
            process=Process.sequential  # This ensures no manager_llm is needed
        )
        
        final_result_string = crew.kickoff()
        
        print(f"--- DEBUG: RAW Output from final (ranker) agent for {resume_name} ---")
        print(final_result_string)
        print("-----------------------------------------------------------")

        task_outputs = extract_json(final_result_string)
        task_outputs['resume_name'] = resume_name
        
        expected_keys = [
            'resume_name', 'overall_score', 'summary', 'justification', 
            'matched_skills', 'missing_skills', 'skills_proficiency_details',
            'technical_skills', 'soft_skills', 'experience', 'education'
        ]
        for key in expected_keys:
            if key not in task_outputs:
                if key == 'overall_score':
                    task_outputs[key] = 0
                elif key in ['summary', 'justification', 'resume_name']:
                    task_outputs[key] = "N/A" if key != 'resume_name' else resume_name
                else:
                    task_outputs[key] = [] 

        print(f"--- DEBUG: Final processed JSON for {resume_name} ---")
        print(task_outputs)
        print("-----------------------------------------------------")
        
        return task_outputs

    except Exception as e:
        print(f"--- ERROR: Unhandled exception in analyze_resume_with_crew for {resume_name}: {e}")
        return {
            "resume_name": resume_name,
            "overall_score": 0,
            "summary": "Error during analysis.",
            "justification": str(e), 
            "matched_skills": [],
            "missing_skills": [],
            "skills_proficiency_details": [],
            "technical_skills": [],
            "soft_skills": [],
            "experience": [],
            "education": []
        }

# -----------------------------
# ASYNC MULTI-RESUME HANDLER
# -----------------------------
async def rank_resumes(jd_text, upload_files):
    """
    Parses and ranks multiple resumes against a job description.
    """
    tasks = []
    for f in upload_files:
        resume_name = getattr(f, "name", "unknown_resume")
        resume_text = parse_resume_file(f)
        
        if not resume_text:
            print(f"--- SKIPPING {resume_name}, could not parse text. ---")
            continue
            
        tasks.append(asyncio.to_thread(analyze_resume_with_crew, jd_text, resume_text, resume_name))
    
    results = await asyncio.gather(*tasks)
    
    return results