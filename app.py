# app.py
import streamlit as st
import pandas as pd
from resume_agent import rank_resumes, parse_resume_file

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>📄 Smart Resume Analyzer & Screener</h1>
    <p style='text-align: center; color: #6c757d;'>Upload your Job Description and candidate resumes to get a ranked analysis.</p>
    """, 
    unsafe_allow_html=True
)

# -----------------------------
# Job Description Upload
# -----------------------------
st.header("📑 Upload Job Description")
jd_file = st.file_uploader("Upload JD (PDF or DOCX)", type=["pdf", "docx"])
jd_text = ""
if jd_file:
    jd_text = parse_resume_file(jd_file)
    st.success("✅ Job Description uploaded successfully!")

# -----------------------------
# Resume Upload
# -----------------------------
st.header("📄 Upload Candidate Resumes")
resume_files = st.file_uploader(
    "Upload Resumes (PDF or DOCX)", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Resumes") and jd_text and resume_files:
    with st.spinner("🔍 Analyzing resumes..."):
        results = rank_resumes(jd_text, resume_files)
    st.success("🎉 Analysis completed! Resumes ranked by overall score:")

    # -----------------------------
    # Helper: Get score badge
    # -----------------------------
    def score_badge(score):
        if score >= 75:
            color = "#4CAF50"  # green
        elif score >= 50:
            color = "#FFC107"  # yellow
        else:
            color = "#F44336"  # red
        return f"<span style='background-color:{color}; color:white; padding:3px 10px; border-radius:5px;'>{score}</span>"

    # -----------------------------
    # Ranking Table with colored scores
    # -----------------------------
    st.subheader("🏆 Ranked Resumes")
    ranking_data = []
    for idx, r in enumerate(results, start=1):
        ranking_data.append({
            "Rank": idx,
            "Resume Name": r["resume_name"],
            "Overall Score": score_badge(r["overall_score"]),
            "Matched Skills": ", ".join(r.get("matched_skills", [])),
            "Missing Skills": ", ".join(r.get("missing_skills", []))
        })
    df_ranked = pd.DataFrame(ranking_data)
    st.write(df_ranked.to_html(escape=False, index=False), unsafe_allow_html=True)

    # -----------------------------
    # Detailed Resume Analysis with Expanders
    # -----------------------------
    st.subheader("🔎 Detailed Resume Analysis")
    for r in results:
        with st.expander(f"📝 {r['resume_name']} (Score: {r['overall_score']})", expanded=False):
            st.markdown(f"**Proficiency Summary:** {r['proficiency_summary']}")
            st.markdown(f"**Justification:** {r['justification']}")

            # Skill-wise proficiency table
            if r.get("skills_detail"):
                st.markdown("**Skill-wise Proficiency:**")
                df_skills = pd.DataFrame(r["skills_detail"])
                st.table(df_skills)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: #6c757d;'>Developed by Kiran Kumar</p>
    """,
    unsafe_allow_html=True
)
