"""
Streamlit UI for the CV ranking system.
"""
import os
import json
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
import tempfile

from .models import JobDescription, CV, MatchingResult, ProcessingStatus
from .job_processor import JobDescriptionProcessor
from .cv_processor import CVProcessor
from .vector_store import VectorStore
from .matching_engine import MatchingEngine


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.job_processor = JobDescriptionProcessor()
        self.cv_processor = CVProcessor()
        self.vector_store = VectorStore()
        self.matching_engine = MatchingEngine()
        
        # Initialize session state
        if 'processed_job' not in st.session_state:
            st.session_state.processed_job = None
        if 'processed_cvs' not in st.session_state:
            st.session_state.processed_cvs = []
        if 'matching_results' not in st.session_state:
            st.session_state.matching_results = []
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = None
    
    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(
            page_title="AI CV Ranking System",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("🤖 AI-Powered CV Ranking System")
        st.markdown("**Streamline your recruitment process with intelligent CV matching**")
        
        # Sidebar for settings
        self._render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Job Description", "📄 CV Processing", "🎯 Matching Results", "📊 Analytics"])
        
        with tab1:
            self._render_job_description_tab()
        
        with tab2:
            self._render_cv_processing_tab()
        
        with tab3:
            self._render_matching_results_tab()
        
        with tab4:
            self._render_analytics_tab()
    
    def _render_sidebar(self):
        """Render the sidebar with settings and status."""
        st.sidebar.header("⚙️ Settings")
        
        # vLLM connection status
        st.sidebar.subheader("🔗 vLLM Connection")
        if self._check_vllm_connection():
            st.sidebar.success("✅ vLLM Connected")
        else:
            st.sidebar.error("❌ vLLM Not Connected")
            st.sidebar.info("Please start vLLM server using: `./run_vllm.sh`")
        
        # Vector store stats
        st.sidebar.subheader("📊 Vector Store")
        stats = self.vector_store.get_collection_stats()
        st.sidebar.metric("CVs Stored", stats["cv_count"])
        st.sidebar.metric("Jobs Stored", stats["job_count"])
        
        # Clear data button
        if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
            self.vector_store.clear_all_data()
            st.session_state.processed_job = None
            st.session_state.processed_cvs = []
            st.session_state.matching_results = []
            st.sidebar.success("Data cleared!")
            st.rerun()
    
    def _check_vllm_connection(self) -> bool:
        """Check if vLLM server is running."""
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _render_job_description_tab(self):
        """Render the job description input tab."""
        st.header("📝 Job Description Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter Job Description")
            
            # Job title input
            job_title = st.text_input(
                "Job Title",
                placeholder="e.g., Senior Python Developer",
                help="Enter the job title"
            )
            
            # Job description text area
            job_description = st.text_area(
                "Job Description",
                height=300,
                placeholder="Paste the complete job description here...",
                help="Include requirements, responsibilities, and qualifications"
            )
            
            # Process button
            if st.button("🔍 Process Job Description", type="primary", disabled=not (job_title and job_description)):
                with st.spinner("Processing job description..."):
                    try:
                        processed_job = self.job_processor.process_job_description(job_title, job_description)
                        st.session_state.processed_job = processed_job
                        
                        # Store in vector database
                        job_id = self.vector_store.add_job_description(processed_job)
                        st.success(f"✅ Job description processed and stored (ID: {job_id[:8]}...)")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing job description: {str(e)}")
        
        with col2:
            st.subheader("📋 Extracted Information")
            
            if st.session_state.processed_job:
                job = st.session_state.processed_job
                
                st.markdown(f"**Title:** {job.title}")
                st.markdown(f"**Experience Required:** {job.experience_years or 'Not specified'} years")
                st.markdown(f"**Education Level:** {job.education_level or 'Not specified'}")
                
                if job.required_skills:
                    st.markdown("**Required Skills:**")
                    for skill in job.required_skills[:5]:  # Show first 5
                        st.markdown(f"• {skill}")
                    if len(job.required_skills) > 5:
                        st.markdown(f"... and {len(job.required_skills) - 5} more")
                
                if job.preferred_skills:
                    st.markdown("**Preferred Skills:**")
                    for skill in job.preferred_skills[:3]:  # Show first 3
                        st.markdown(f"• {skill}")
                    if len(job.preferred_skills) > 3:
                        st.markdown(f"... and {len(job.preferred_skills) - 3} more")
                
                st.markdown(f"**Keywords Extracted:** {len(job.keywords)}")
            else:
                st.info("👆 Process a job description to see extracted information")
    
    def _render_cv_processing_tab(self):
        """Render the CV processing tab."""
        st.header("📄 CV Processing")
        
        if not st.session_state.processed_job:
            st.warning("⚠️ Please process a job description first before uploading CVs.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📁 Upload CVs")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Choose CV files (PDF only)",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload multiple PDF CV files (max 50)"
            )
            
            if uploaded_files:
                # Limit to 50 files
                if len(uploaded_files) > 50:
                    st.warning(f"⚠️ Limiting to first 50 files")
                    uploaded_files = uploaded_files[:50]
                
                st.info(f"📄 {len(uploaded_files)} file(s) selected")
                
                # Process button
                if st.button("🔄 Process CVs", type="primary"):
                    self._process_uploaded_cvs(uploaded_files)
        
        with col2:
            st.subheader("📊 Processing Status")
            
            if st.session_state.processing_status:
                status = st.session_state.processing_status
                st.progress(status.progress)
                st.markdown(f"**Stage:** {status.stage}")
                st.markdown(f"**Message:** {status.message}")
                
                if status.details:
                    for key, value in status.details.items():
                        st.markdown(f"**{key}:** {value}")
            
            # Show processed CVs
            if st.session_state.processed_cvs:
                st.subheader("✅ Processed CVs")
                for i, cv in enumerate(st.session_state.processed_cvs):
                    with st.expander(f"📄 {cv.filename}"):
                        st.markdown(f"**Name:** {cv.name or 'Not found'}")
                        st.markdown(f"**Email:** {cv.email or 'Not found'}")
                        st.markdown(f"**Experience:** {cv.total_experience_years or 0} years")
                        st.markdown(f"**Skills:** {len(cv.skills)} found")
                        st.markdown(f"**Education:** {len(cv.education)} entries")
    
    def _process_uploaded_cvs(self, uploaded_files):
        """Process uploaded CV files."""
        processed_cvs = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update status
            progress = (i + 1) / len(uploaded_files)
            st.session_state.processing_status = ProcessingStatus(
                stage="Processing CVs",
                progress=progress,
                message=f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})",
                details={"Current file": uploaded_file.name}
            )
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process CV
                cv = self.cv_processor.process_cv(tmp_path)
                processed_cvs.append(cv)
                
                # Store in vector database
                cv_id = self.vector_store.add_cv(cv)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        # Update session state
        st.session_state.processed_cvs = processed_cvs
        st.session_state.processing_status = ProcessingStatus(
            stage="Completed",
            progress=1.0,
            message=f"Successfully processed {len(processed_cvs)} CVs",
            details={"Total CVs": len(processed_cvs)}
        )
        
        st.success(f"✅ Successfully processed {len(processed_cvs)} CVs!")
        st.rerun()
    
    def _render_matching_results_tab(self):
        """Render the matching results tab."""
        st.header("🎯 CV Matching Results")
        
        if not st.session_state.processed_job:
            st.warning("⚠️ Please process a job description first.")
            return
        
        if not st.session_state.processed_cvs:
            st.warning("⚠️ Please process some CVs first.")
            return
        
        # Match button
        if st.button("🎯 Run Matching Analysis", type="primary"):
            self._run_matching_analysis()
        
        # Display results
        if st.session_state.matching_results:
            self._display_matching_results()
    
    def _run_matching_analysis(self):
        """Run the matching analysis."""
        with st.spinner("🔍 Running matching analysis..."):
            try:
                job = st.session_state.processed_job
                cvs = st.session_state.processed_cvs
                
                matching_results = []
                
                for cv in cvs:
                    result = self.matching_engine.match_cv_to_job(cv, job)
                    matching_results.append(result)
                
                # Rank results
                ranked_results = self.matching_engine.rank_cvs(matching_results)
                st.session_state.matching_results = ranked_results
                
                st.success(f"✅ Matching analysis completed for {len(ranked_results)} CVs!")
                
            except Exception as e:
                st.error(f"❌ Error running matching analysis: {str(e)}")
    
    def _display_matching_results(self):
        """Display the matching results."""
        results = st.session_state.matching_results
        
        st.subheader(f"📊 Results ({len(results)} CVs)")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total CVs", len(results))
        
        with col2:
            excellent_matches = len([r for r in results if r.total_score >= 0.8])
            st.metric("Excellent Matches", excellent_matches)
        
        with col3:
            good_matches = len([r for r in results if 0.6 <= r.total_score < 0.8])
            st.metric("Good Matches", good_matches)
        
        with col4:
            avg_score = sum(r.total_score for r in results) / len(results)
            st.metric("Average Score", f"{avg_score:.1%}")
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        # Create results dataframe
        results_data = []
        for result in results:
            results_data.append({
                "Rank": result.rank,
                "CV File": result.cv_filename,
                "Candidate": result.candidate_name or "Unknown",
                "Overall Score": f"{result.total_score:.1%}",
                "Summary": result.summary
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
        
        # Individual result details
        st.subheader("🔍 Individual Results")
        
        for result in results:
            with st.expander(f"#{result.rank} {result.cv_filename} - {result.total_score:.1%}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**Candidate:** {result.candidate_name or 'Unknown'}")
                    st.markdown(f"**Overall Score:** {result.total_score:.1%}")
                    st.markdown(f"**Summary:** {result.summary}")
                
                with col2:
                    st.markdown("**Component Scores:**")
                    for score in result.individual_scores:
                        st.markdown(f"• **{score.component.title()}:** {score.score:.1%}")
                        st.markdown(f"  - {score.details}")
                        if score.matched_items:
                            st.markdown(f"  - Matched: {', '.join(score.matched_items[:3])}")
                            if len(score.matched_items) > 3:
                                st.markdown(f"  - ... and {len(score.matched_items) - 3} more")
        
        # Export functionality
        self._render_export_section()
    
    def _render_export_section(self):
        """Render the export functionality."""
        st.subheader("📤 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to CSV", type="secondary"):
                self._export_to_csv()
        
        with col2:
            if st.button("📄 Export to JSON", type="secondary"):
                self._export_to_json()
    
    def _export_to_csv(self):
        """Export results to CSV."""
        try:
            results = st.session_state.matching_results
            job = st.session_state.processed_job
            
            # Create detailed CSV data
            csv_data = []
            for result in results:
                row = {
                    "Rank": result.rank,
                    "CV_Filename": result.cv_filename,
                    "Candidate_Name": result.candidate_name or "Unknown",
                    "Overall_Score": result.total_score,
                    "Summary": result.summary
                }
                
                # Add individual component scores
                for score in result.individual_scores:
                    row[f"{score.component}_score"] = score.score
                    row[f"{score.component}_details"] = score.details
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cv_matching_results_{timestamp}.csv"
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error exporting to CSV: {str(e)}")
    
    def _export_to_json(self):
        """Export results to JSON."""
        try:
            results = st.session_state.matching_results
            job = st.session_state.processed_job
            
            # Prepare export data
            export_data = {
                "job_description": {
                    "title": job.title,
                    "experience_years": job.experience_years,
                    "education_level": job.education_level,
                    "required_skills": job.required_skills,
                    "preferred_skills": job.preferred_skills
                },
                "matching_results": [
                    {
                        "rank": result.rank,
                        "cv_filename": result.cv_filename,
                        "candidate_name": result.candidate_name,
                        "total_score": result.total_score,
                        "summary": result.summary,
                        "individual_scores": [
                            {
                                "component": score.component,
                                "score": score.score,
                                "details": score.details,
                                "matched_items": score.matched_items
                            }
                            for score in result.individual_scores
                        ]
                    }
                    for result in results
                ],
                "export_timestamp": datetime.now().isoformat(),
                "total_cvs": len(results)
            }
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cv_matching_results_{timestamp}.json"
            
            # Download button
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error exporting to JSON: {str(e)}")
    
    def _render_analytics_tab(self):
        """Render the analytics tab."""
        st.header("📊 Analytics & Insights")
        
        if not st.session_state.matching_results:
            st.info("👆 Run matching analysis to see analytics")
            return
        
        results = st.session_state.matching_results
        
        # Score distribution
        st.subheader("📈 Score Distribution")
        
        scores = [r.total_score for r in results]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Matching Score')
                ax.set_ylabel('Number of CVs')
                ax.set_title('Distribution of Matching Scores')
                st.pyplot(fig)
            except ImportError:
                st.error("Matplotlib not available for plotting. Please install matplotlib.")
        
        with col2:
            # Statistics
            st.markdown("**Statistics:**")
            st.markdown(f"• Mean Score: {sum(scores)/len(scores):.1%}")
            st.markdown(f"• Median Score: {sorted(scores)[len(scores)//2]:.1%}")
            st.markdown(f"• Highest Score: {max(scores):.1%}")
            st.markdown(f"• Lowest Score: {min(scores):.1%}")
        
        # Component analysis
        st.subheader("🔍 Component Analysis")
        
        component_scores = {}
        for result in results:
            for score in result.individual_scores:
                if score.component not in component_scores:
                    component_scores[score.component] = []
                component_scores[score.component].append(score.score)
        
        # Component comparison chart
        if component_scores:
            component_means = {comp: sum(scores)/len(scores) for comp, scores in component_scores.items()}
            
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                components = list(component_means.keys())
                means = list(component_means.values())
                
                bars = ax.bar(components, means, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                ax.set_ylabel('Average Score')
                ax.set_title('Average Scores by Component')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, mean in zip(bars, means):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{mean:.1%}', ha='center', va='bottom')
                
                st.pyplot(fig)
            except ImportError:
                st.error("Matplotlib not available for plotting. Please install matplotlib.")
        
        # Top performers
        st.subheader("🏆 Top Performers")
        
        top_3 = results[:3]
        for i, result in enumerate(top_3):
            st.markdown(f"**#{i+1} {result.cv_filename}** - {result.total_score:.1%}")
            st.markdown(f"*{result.summary}*")
            st.markdown("---")
