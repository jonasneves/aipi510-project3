"""
Streamlit frontend for AI Salary Prediction.

Run with: streamlit run frontend/app.py
"""

import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="centered",
)

# API endpoint (configurable)
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="URL of the prediction API",
)

st.title("AI/ML Salary Predictor")
st.markdown("Predict salaries for AI and ML roles based on job details.")

# Input form
st.header("Job Details")

col1, col2 = st.columns(2)

with col1:
    job_title = st.text_input(
        "Job Title",
        value="ML Engineer",
        placeholder="e.g., Senior Data Scientist",
    )
    location = st.selectbox(
        "Location (State)",
        options=["CA", "NY", "WA", "TX", "MA", "CO", "IL", "GA", "NC", "FL", "PA", "VA", "AZ", "OR", "MD"],
        index=0,
    )
    experience = st.slider(
        "Years of Experience",
        min_value=0,
        max_value=20,
        value=3,
    )

with col2:
    company = st.text_input(
        "Company (optional)",
        value="",
        placeholder="e.g., Google",
    )
    skills_input = st.text_area(
        "Skills (comma-separated)",
        value="python, machine learning",
        placeholder="e.g., pytorch, nlp, kubernetes",
        height=100,
    )

# Parse skills
skills = [s.strip() for s in skills_input.split(",") if s.strip()]

# Predict button
if st.button("Predict Salary", type="primary"):
    # Prepare request
    payload = {
        "job_title": job_title,
        "location": location,
        "experience_years": experience,
        "company": company if company else None,
        "skills": skills if skills else None,
    }

    try:
        with st.spinner("Predicting..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

        if response.status_code == 200:
            result = response.json()

            st.success("Prediction Complete")

            # Display results
            st.header("Predicted Salary")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Low (90% CI)", f"${result['salary_low']:,}")
            with col2:
                st.metric("Predicted", f"${result['predicted_salary']:,}")
            with col3:
                st.metric("High (90% CI)", f"${result['salary_high']:,}")

            # Summary
            st.markdown("---")
            st.subheader("Summary")
            st.markdown(f"""
            - **Job Title:** {job_title}
            - **Location:** {location}
            - **Experience:** {experience} years
            - **Company:** {company if company else "Not specified"}
            - **Skills:** {", ".join(skills) if skills else "Not specified"}
            """)

        elif response.status_code == 503:
            st.error("Model not loaded. Please ensure the API is running and the model is trained.")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to API at {API_URL}. Make sure the API is running.")
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    """
    This app predicts AI/ML salaries based on:
    - Job title and seniority
    - Location (US states)
    - Years of experience
    - Company tier
    - Technical skills

    Data sources: H1B filings, BLS statistics, job postings.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### API Status")
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.warning("API Degraded")
except Exception:
    st.sidebar.error("API Offline")
