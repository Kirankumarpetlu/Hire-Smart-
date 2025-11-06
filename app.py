import streamlit as st
import asyncio
import json
import pandas as pd
from resume_agent_test import rank_resumes
from pdfminer.high_level import extract_text
from docx import Document

st.set_page_config(
    page_title="Hire Smart ",
    layout="wide",
    page_icon="💼",
)

st.title("🤖 Hire Smart")
st.markdown(
    "This system evaluates multiple resumes against a job description using a **Crew of AI Agents** powered by a fine-tuned Large Language Model."
)
st.markdown("---")

# ==============================
# Upload Section
# ==============================
col1, col2 = st.columns(2)

with col1:
    jd_file = st.file_uploader("📄 Upload Job Description (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"])

with col2:
    resume_files = st.file_uploader(
        "👥 Upload Candidate Resumes (Multiple files supported)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

if st.button("🚀 Analyze & Rank Candidates"):
    if not jd_file or not resume_files:
        st.warning("Please upload both Job Description and at least one Resume.")
    else:
        st.info("⏳ Processing with AI Agents... This may take a few minutes depending on the number of resumes.")

        # ==============================
        # Extract JD text
        # ==============================
        def read_text(file):
            name = file.name.lower()
            if name.endswith(".pdf"):
                return extract_text(file)
            elif name.endswith(".docx"):
                doc = Document(file)
                return "\n".join([p.text for p in doc.paragraphs])
            else:
                return file.read().decode("utf-8", errors="ignore")

        jd_text = read_text(jd_file)

        # ==============================
        # Run AI Pipeline
        # ==============================
        async def process_pipeline():
            results_data = await rank_resumes(jd_text, resume_files)
            return results_data

        # 'results' is now the list directly
        results_list = asyncio.run(process_pipeline())

        # Sort candidates by score
        # `rank_resumes` returns a dict {"results": [...], "metrics": {...}}
        # but older code expected a plain list. Handle both shapes robustly.
        if isinstance(results_list, dict):
            candidates = results_list.get("results", [])
        elif isinstance(results_list, list):
            candidates = results_list
        else:
            candidates = []

        sorted_results = sorted(
            candidates,
            key=lambda x: x.get("overall_score", 0) if isinstance(x, dict) else 0,
            reverse=True,
        )

        # Add rank column
        for i, r in enumerate(sorted_results, 1):
            r["rank"] = i

        # ==============================
        # Display Results
        # ==============================
        st.markdown("## 🧠 Candidate Evaluation Results")

        # Show summary table
        table_data = [
            {
                "Rank": r["rank"],
                "Candidate": r["resume_name"],
                "Score": r.get("overall_score", 0),
                "Summary": r.get("summary", ""),
            }
            for r in sorted_results
        ]
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Expanders for detailed views
        for res in sorted_results:
            score = res.get('overall_score', 0)
            with st.expander(f"👤 Rank {res['rank']} — {res['resume_name']} (Score: {score})"):
                
                st.markdown(f"**Summary:** {res.get('summary', 'No summary provided.')}")
                st.markdown(f"**Justification:** {res.get('justification', 'No justification provided.')}")

                # <<< --- UPDATED SECTION --- >>>
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### ✅ Matched Skills:")
                    st.write(res.get("matched_skills", []))
                with col2:
                    st.markdown("### ❌ Missing Skills:")
                    st.write(res.get("missing_skills", []))

                st.markdown("### 📊 Skill Proficiency Details:")
                st.write(res.get("skills_proficiency_details", []))
                
                # <<< --- ADD THIS ENTIRE NEW SECTION --- >>>
                st.markdown("---") # Visual separator
                
                st.markdown(f"### 🔎 Agent-Parsed Details for: {res['resume_name']}")
                
                tab1, tab2, tab3, tab4 = st.tabs(["Skills", "Experience", "Education", "Full JSON"])

                with tab1:
                    st.markdown("#### Technical Skills")
                    st.write(res.get("technical_skills", []))
                    st.markdown("#### Soft Skills")
                    st.write(res.get("soft_skills", []))
                
                with tab2:
                    st.markdown("#### Parsed Experience")
                    st.write(res.get("experience", []))
                
                with tab3:
                    st.markdown("#### Parsed Education")
                    st.write(res.get("education", []))
                
                with tab4:
                    st.markdown("#### Full Analyzed JSON")
                    st.json(res)
                # <<< --- END OF NEW SECTION --- >>>
                
        # ==============================
        # Optional Chart Visualization
        # ==============================
        try:
            import plotly.express as px

            fig = px.bar(
                df,
                x="Candidate",
                y="Score",
                color="Score",
                text="Rank",
                title="Candidate Ranking Visualization",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart visualization skipped: {e}")

        st.success("✅ Analysis complete! Check detailed results above.")

        st.markdown("""
            <hr>
            <p style='text-align:center; font-size:14px; color:gray;'>
                Developed by <b>Kiran Kumar </b> 
            </p>
        """, unsafe_allow_html=True)