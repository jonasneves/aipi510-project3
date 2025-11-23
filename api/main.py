"""
FastAPI application for AI/ML salary prediction service.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import SalaryPredictor
from src.processing import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Salary Prediction API",
    description="Predict AI/ML salaries based on job title, location, experience, and skills",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class SalaryResponse(BaseModel):
    """Response model for salary predictions."""

    predicted_salary: int
    salary_low: int
    salary_high: int
    confidence_level: str


class BatchSalaryRequest(BaseModel):
    """Request model for batch predictions."""

    jobs: list[SalaryInput]


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
    data = engineer.encode_categoricals(data, fit=True)

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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Salary Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict salary for a single job",
            "/predict/batch": "POST - Predict salaries for multiple jobs",
            "/health": "GET - Health check",
            "/model/info": "GET - Model information",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None,
    }


@app.get("/badge/api")
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


@app.get("/badge/app")
async def badge_app():
    """Shields.io badge endpoint for App status (proxied via API)."""
    return {
        "schemaVersion": 1,
        "label": "App",
        "message": "online",
        "color": "brightgreen",
    }


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": "XGBoost Regressor",
        "n_features": len(predictor.feature_names),
        "training_metrics": predictor.training_metrics.get("train", {}),
    }


@app.get("/options")
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


@app.post("/predict", response_model=SalaryResponse)
async def predict(job: SalaryInput):
    """Predict salary for a single job."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = prepare_features(job)
        result = predictor.predict_with_range(X)

        return SalaryResponse(
            predicted_salary=int(result["predicted_salary"].iloc[0]),
            salary_low=int(result["salary_low"].iloc[0]),
            salary_high=int(result["salary_high"].iloc[0]),
            confidence_level="90%",
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchSalaryRequest):
    """Predict salaries for multiple jobs."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []
        for job in request.jobs:
            result = await predict(job)
            predictions.append(result.model_dump())

        return {"predictions": predictions, "count": len(predictions)}

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
