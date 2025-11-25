"""
FastAPI application for AI/ML salary prediction service.
"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Developed the FastAPI service architecture, resume parsing functionality, feature engineering
## pipeline, and prediction endpoints with confidence intervals and factor analysis.

import io
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SalaryPredictor
from src.processing import FeatureEngineer
from src.utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router for API endpoints (will be mounted at /api)
router = APIRouter()

# Load patterns and config
_patterns_config = ConfigLoader.get_patterns()
_features_config = ConfigLoader.get_features()

JOB_TITLE_PATTERNS = [
    (item["pattern"], item["display"])
    for item in _patterns_config["job_title_patterns"]
]
SKILL_PATTERNS = _patterns_config["skill_patterns"]
STATE_PATTERNS = _patterns_config["state_patterns"]

_salary_factors = _patterns_config["salary_factors"]
HIGH_PAY_STATES = _salary_factors["high_pay_states"]
LOW_PAY_STATES = _salary_factors["low_pay_states"]
HIGH_VALUE_SKILLS = _salary_factors["high_value_skills"]
SENIOR_KEYWORDS = _salary_factors["senior_keywords"]

_encodings = _features_config["encodings"]
STATE_INDEX = _encodings["state_index"]
TIER_INDEX = _encodings["tier_index"]


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF file."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


def parse_resume(text: str) -> dict:
    """Parse resume text to extract job-related information."""
    text_lower = text.lower()

    # Extract job title
    job_title = None
    for pattern, title in JOB_TITLE_PATTERNS:
        if re.search(pattern, text_lower):
            job_title = title
            break

    # Extract skills
    skills = []
    for skill, pattern in SKILL_PATTERNS.items():
        if re.search(pattern, text_lower):
            skills.append(skill)

    # Extract location
    location = None
    for state, pattern in STATE_PATTERNS.items():
        if re.search(pattern, text_lower):
            location = state
            break

    # Extract years of experience
    experience_years = 3  # default
    yoe_patterns = [
        r"(\d+)\+?\s*years?\s*(of)?\s*(professional)?\s*experience",
        r"experience[:\s]+(\d+)\+?\s*years?",
        r"(\d+)\+?\s*years?\s*in\s*(ml|ai|data|software)",
    ]
    for pattern in yoe_patterns:
        match = re.search(pattern, text_lower)
        if match:
            experience_years = min(int(match.group(1)), 30)
            break

    return {
        "job_title": job_title,
        "location": location,
        "experience_years": experience_years,
        "skills": skills,
        "extracted_text_preview": text[:500] if text else None,
    }


app = FastAPI(
    title="AI Salary Prediction API",
    description="Predict AI/ML salaries based on job title, location, experience, and skills",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root-level health check for internal monitoring (workflow checks localhost:8000/health)
@app.get("/health")
async def root_health():
    """Health check at root level for internal monitoring."""
    return {"status": "healthy", "model_loaded": predictor is not None}


# Global variables
predictor: Optional[SalaryPredictor] = None
engineer: Optional[FeatureEngineer] = None


class SalaryInput(BaseModel):
    """Input features for salary prediction."""

    job_title: str = Field(..., description="Job title (e.g., 'ML Engineer', 'Data Scientist')")
    location: str = Field(..., description="US state abbreviation (e.g., 'CA', 'NY')")
    experience_years: int = Field(3, description="Years of experience", ge=0, le=30)
    company: Optional[str] = Field(None, description="Company name (optional)")
    skills: Optional[list[str]] = Field(None, description="List of skills (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "job_title": "Senior ML Engineer",
                "location": "CA",
                "experience_years": 5,
                "company": "Google",
                "skills": ["pytorch", "nlp", "kubernetes"],
            }
        }


class FeatureFactor(BaseModel):
    """A single feature factor affecting salary."""

    name: str
    impact: str  # "positive" or "negative"
    description: str


class SalaryResponse(BaseModel):
    """Response model for salary predictions."""

    predicted_salary: int
    salary_low: int
    salary_high: int
    confidence_level: str
    top_factors: list[FeatureFactor] = []


class FeatureImportanceResponse(BaseModel):
    """Response model for feature importance."""

    features: list[dict]
    total_features: int




def load_model():
    """Load the trained model."""
    global predictor, engineer

    model_path = Path("models/salary_model_latest.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    predictor = SalaryPredictor(model_dir="models")
    predictor.load()

    engineer = FeatureEngineer()

    logger.info("Model loaded successfully")


def prepare_features(input_data: SalaryInput) -> pd.DataFrame:
    """Prepare features for prediction."""
    data = pd.DataFrame(
        [
            {
                "job_title": input_data.job_title,
                "employer_name": input_data.company,
                "worksite_state": input_data.location.upper(),
            }
        ]
    )

    data = engineer.engineer_features(data)

    # Don't use fit=True for categoricals - it creates new encoders each time
    # Instead, manually create state encoding based on known states
    state = input_data.location.upper()

    data["state_clean_encoded"] = STATE_INDEX.get(state, 20)

    company_tier = data["company_tier"].iloc[0] if "company_tier" in data.columns else "unknown"
    data["company_tier_encoded"] = TIER_INDEX.get(company_tier, 6)

    if "estimated_yoe" in data.columns:
        data["estimated_yoe"] = input_data.experience_years

    if input_data.skills:
        skills_lower = [s.lower().strip() for s in input_data.skills]
        for col in data.columns:
            if col.startswith("skill_"):
                skill_name = col.replace("skill_", "").replace("_", " ")
                if any(skill_name in s or s in skill_name for s in skills_lower):
                    data[col] = 1

    unique_features = list(dict.fromkeys(predictor.feature_names))
    X = pd.DataFrame(index=[0])
    for col in unique_features:
        if col in data.columns:
            val = data[col].iloc[0]
            X[col] = pd.to_numeric(val, errors="coerce") if not isinstance(val, (int, float)) else val
        else:
            X[col] = 0
    X = X.fillna(0)

    return X


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Salary Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/api/predict": "POST - Predict salary for a single job",
            "/api/predict/batch": "POST - Predict salaries for multiple jobs",
            "/api/health": "GET - Health check",
            "/api/model/info": "GET - Model information",
        },
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
    }


@router.get("/badge/api")
async def badge_api():
    """Shields.io badge endpoint for API status."""
    if predictor is not None:
        return {
            "schemaVersion": 1,
            "label": "API",
            "message": "online",
            "color": "brightgreen",
        }
    return {
        "schemaVersion": 1,
        "label": "API",
        "message": "degraded",
        "color": "yellow",
    }


@router.get("/badge/app")
async def badge_app():
    """Shields.io badge endpoint for App status (proxied via API)."""
    return {
        "schemaVersion": 1,
        "label": "App",
        "message": "online",
        "color": "brightgreen",
    }


@router.get("/model/info")
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "XGBoost Regressor",
        "n_features": len(predictor.feature_names),
        "feature_names": predictor.feature_names[:50],  # First 50 features for debugging
        "training_metrics": predictor.training_metrics.get("train", {}),
    }


@router.get("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(top_n: int = 20):
    """Get global feature importance from the model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if predictor.feature_importance is None:
        raise HTTPException(status_code=503, detail="Feature importance not available")

    # Sort features by importance
    sorted_features = sorted(
        predictor.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    # Format feature names for display
    features = []
    for name, importance in sorted_features:
        display_name = format_feature_name(name)
        features.append({
            "name": name,
            "display_name": display_name,
            "importance": round(importance, 4),
            "importance_pct": round(importance * 100, 2)
        })

    return FeatureImportanceResponse(
        features=features,
        total_features=len(predictor.feature_names)
    )


def format_feature_name(name: str) -> str:
    """Format internal feature name to human-readable display name."""
    # Handle common patterns
    if name == "estimated_yoe":
        return "Years of Experience"
    if name == "state_clean_encoded":
        return "Location (State)"
    if name == "company_tier_encoded":
        return "Company Tier"
    if name.startswith("skill_"):
        skill = name.replace("skill_", "").replace("_", " ").title()
        return f"Skill: {skill}"
    if name.startswith("title_"):
        return "Job Title Level"
    if name == "is_senior":
        return "Senior Level"
    if name == "is_staff_principal":
        return "Staff/Principal Level"
    if name == "is_manager_director":
        return "Manager/Director Level"

    # Default: clean up underscores and title case
    return name.replace("_", " ").title()


def calculate_top_factors(input_data: SalaryInput, X: pd.DataFrame) -> list[FeatureFactor]:
    """Calculate top factors affecting this specific prediction."""
    if predictor is None or predictor.feature_importance is None:
        return []

    factors = []

    # Get top important features from model
    sorted_importance = sorted(
        predictor.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Map user inputs to features and their importance
    # Experience
    if "estimated_yoe" in predictor.feature_importance:
        exp_importance = predictor.feature_importance["estimated_yoe"]
        if input_data.experience_years >= 7:
            factors.append(FeatureFactor(
                name="Experience",
                impact="positive",
                description=f"{input_data.experience_years} years (senior level)"
            ))
        elif input_data.experience_years >= 4:
            factors.append(FeatureFactor(
                name="Experience",
                impact="positive",
                description=f"{input_data.experience_years} years (mid level)"
            ))
        elif input_data.experience_years <= 2:
            factors.append(FeatureFactor(
                name="Experience",
                impact="negative",
                description=f"{input_data.experience_years} years (entry level)"
            ))

    # Location
    if input_data.location.upper() in HIGH_PAY_STATES:
        factors.append(FeatureFactor(
            name="Location",
            impact="positive",
            description=f"{input_data.location} (high-paying market)"
        ))
    elif input_data.location.upper() in LOW_PAY_STATES:
        factors.append(FeatureFactor(
            name="Location",
            impact="negative",
            description=f"{input_data.location} (lower cost market)"
        ))

    # Job title seniority
    title_lower = input_data.job_title.lower()
    if any(x in title_lower for x in SENIOR_KEYWORDS):
        factors.append(FeatureFactor(
            name="Seniority",
            impact="positive",
            description="Principal/Staff/Director level"
        ))
    elif "senior" in title_lower:
        factors.append(FeatureFactor(
            name="Seniority",
            impact="positive",
            description="Senior level"
        ))
    elif not any(x in title_lower for x in ["senior", "staff", "principal", "lead"]):
        factors.append(FeatureFactor(
            name="Seniority",
            impact="negative",
            description="Entry/Mid level title"
        ))

    # Skills
    if input_data.skills:
        skills_upper = [s.lower() for s in input_data.skills]
        has_high_value = [s for s in HIGH_VALUE_SKILLS if s.lower() in skills_upper]
        if has_high_value:
            factors.append(FeatureFactor(
                name="Skills",
                impact="positive",
                description=f"High-value: {', '.join(has_high_value[:3])}"
            ))

        # Check for missing high-value skills
        missing_skills = [s for s in HIGH_VALUE_SKILLS[:3] if s.lower() not in skills_upper]
        if missing_skills and len(factors) < 4:
            factors.append(FeatureFactor(
                name="Missing Skills",
                impact="negative",
                description=f"Consider adding: {missing_skills[0]}"
            ))
    else:
        factors.append(FeatureFactor(
            name="Skills",
            impact="negative",
            description="No skills specified"
        ))

    # Limit to top 4 factors
    return factors[:4]


@router.get("/options")
async def get_options():
    """Get available options for prediction inputs."""
    return {
        "job_titles": [
            "ML Engineer",
            "Senior ML Engineer",
            "Staff ML Engineer",
            "Principal ML Engineer",
            "Data Scientist",
            "Senior Data Scientist",
            "Staff Data Scientist",
            "AI Engineer",
            "Senior AI Engineer",
            "Research Scientist",
            "Senior Research Scientist",
            "Applied Scientist",
            "Data Engineer",
            "Senior Data Engineer",
            "MLOps Engineer",
            "AI/ML Manager",
            "Director of ML",
            "VP of AI",
        ],
        "locations": [
            {"code": "CA", "name": "California"},
            {"code": "NY", "name": "New York"},
            {"code": "WA", "name": "Washington"},
            {"code": "TX", "name": "Texas"},
            {"code": "MA", "name": "Massachusetts"},
            {"code": "CO", "name": "Colorado"},
            {"code": "IL", "name": "Illinois"},
            {"code": "GA", "name": "Georgia"},
            {"code": "NC", "name": "North Carolina"},
            {"code": "FL", "name": "Florida"},
            {"code": "PA", "name": "Pennsylvania"},
            {"code": "VA", "name": "Virginia"},
            {"code": "AZ", "name": "Arizona"},
            {"code": "OR", "name": "Oregon"},
            {"code": "MD", "name": "Maryland"},
            {"code": "NJ", "name": "New Jersey"},
            {"code": "OH", "name": "Ohio"},
            {"code": "MI", "name": "Michigan"},
            {"code": "MN", "name": "Minnesota"},
            {"code": "UT", "name": "Utah"},
        ],
        "skills": [
            "Python",
            "Machine Learning",
            "Deep Learning",
            "PyTorch",
            "TensorFlow",
            "NLP",
            "Computer Vision",
            "MLOps",
            "Kubernetes",
            "AWS",
            "GCP",
            "SQL",
            "Spark",
            "LLMs",
            "Transformers",
        ],
    }


class ResumeParseResponse(BaseModel):
    """Response model for resume parsing."""

    job_title: Optional[str] = None
    location: Optional[str] = None
    experience_years: int = 3
    skills: list[str] = []
    success: bool = True
    message: str = "Resume parsed successfully"


@router.post("/parse-resume", response_model=ResumeParseResponse)
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """Parse a resume file (PDF or DOCX) to extract job information."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".pdf") or filename_lower.endswith(".docx")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a PDF or DOCX file.",
        )

    try:
        content = await file.read()

        if filename_lower.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        else:
            text = extract_text_from_docx(content)

        if not text.strip():
            return ResumeParseResponse(
                success=False,
                message="Could not extract text from file. Please try a different format.",
            )

        parsed = parse_resume(text)

        return ResumeParseResponse(
            job_title=parsed["job_title"],
            location=parsed["location"],
            experience_years=parsed["experience_years"],
            skills=parsed["skills"],
            success=True,
            message="Resume parsed successfully" if parsed["job_title"] else "Resume parsed but no job title detected",
        )

    except Exception as e:
        logger.error(f"Resume parsing error: {e}")
        return ResumeParseResponse(
            success=False,
            message=f"Error parsing resume: {str(e)}",
        )


@router.post("/predict", response_model=SalaryResponse)
async def predict(job: SalaryInput):
    """Predict salary for a single job."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = prepare_features(job)
        result = predictor.predict_with_range(X)

        # Calculate top factors affecting this prediction
        top_factors = calculate_top_factors(job, X)

        return SalaryResponse(
            predicted_salary=int(result["predicted_salary"].iloc[0]),
            salary_low=int(result["salary_low"].iloc[0]),
            salary_high=int(result["salary_high"].iloc[0]),
            confidence_level="90%",
            top_factors=top_factors,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Mount router AFTER all routes are defined
# (include_router copies routes at call time, so routes must exist first)
app.include_router(router, prefix="/api")
app.include_router(router, prefix="")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
