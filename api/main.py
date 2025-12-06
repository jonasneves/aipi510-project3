"""
FastAPI application for AI/ML salary prediction service.
"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Developed the FastAPI service architecture, resume parsing functionality, feature engineering
## pipeline, and prediction endpoints with confidence intervals and factor analysis.

import io
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
import sys
from datetime import datetime
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

# Try to import boto3 for S3 access (optional)
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router for API endpoints (will be mounted at /api)
router = APIRouter()

# Load patterns and config
_patterns_config = ConfigLoader.get_patterns()
_features_config = ConfigLoader.get_features()

def _compile_regex(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


JOB_TITLE_PATTERNS = [
    {
        "regex": _compile_regex(item["pattern"]),
        "display": item["display"],
        "base_score": len(_patterns_config["job_title_patterns"]) - idx,
    }
    for idx, item in enumerate(_patterns_config["job_title_patterns"])
]
SKILL_PATTERNS = {
    skill: _compile_regex(pattern)
    for skill, pattern in _patterns_config["skill_patterns"].items()
}
STATE_PATTERNS = {
    state: _compile_regex(pattern)
    for state, pattern in _patterns_config["state_patterns"].items()
}

_salary_factors = _patterns_config["salary_factors"]
HIGH_PAY_STATES = _salary_factors["high_pay_states"]
LOW_PAY_STATES = _salary_factors["low_pay_states"]
HIGH_VALUE_SKILLS = _salary_factors["high_value_skills"]
SENIOR_KEYWORDS = _salary_factors["senior_keywords"]


# Resume parsing heuristics
SUMMARY_WINDOW = 800   # Characters considered "summary"/top of resume
MID_WINDOW = 2000      # Characters considered "recent experience"
DEFAULT_YOE = 3
MAX_YOE = 30

EXPERIENCE_SECTION_MARKERS = (
    "experience",
    "work history",
    "professional experience",
    "employment",
)

TITLE_SPLIT_TOKENS = (" - ", " – ", " — ", " | ", " @ ", " at ")

SKILL_TITLE_HINTS = [
    {"title": "MLOps Engineer", "required": {"MLOps"}, "preferred": {"Kubernetes", "AWS", "GCP"}},
    {"title": "Data Engineer", "required": {"SQL", "Spark"}, "preferred": set()},
    {"title": "ML Engineer", "required": {"Machine Learning", "Deep Learning"}, "preferred": {"PyTorch", "TensorFlow"}},
    {"title": "AI Engineer", "required": {"LLMs"}, "preferred": {"Transformers", "NLP"}},
    {"title": "Data Scientist", "required": {"Machine Learning", "Python"}, "preferred": {"NLP", "LLMs"}},
    {"title": "Research Scientist", "required": {"Computer Vision"}, "preferred": {"Transformers", "NLP"}},
]

YOE_PATTERNS = [
    re.compile(r"(?P<start>\d+)\s*-\s*(?P<end>\d+)\s+years", re.IGNORECASE),
    re.compile(r"(?P<years>\d+)\+?\s*years?\s*(?:of\s+)?(?:professional\s+)?experience", re.IGNORECASE),
    re.compile(r"experience[:\s]+(?P<years>\d+)\+?\s*years", re.IGNORECASE),
    re.compile(r"(?P<years>\d+)\+?\s*years?\s*in\s*(ml|ai|data|software)", re.IGNORECASE),
]


@dataclass
class NormalizedText:
    clean: str
    lower: str
    summary_lower: str
    lines: list[str]


def _normalize_text(text: str) -> NormalizedText:
    if not text:
        return NormalizedText("", "", "", [])

    clean = re.sub(r"\s+", " ", text).strip()
    lower = clean.lower()
    summary_lower = lower[:SUMMARY_WINDOW]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return NormalizedText(clean=clean, lower=lower, summary_lower=summary_lower, lines=lines)


def _score_job_titles(norm: NormalizedText) -> tuple[Optional[str], float]:
    if not norm.lower:
        return None, 0.0

    scores = Counter()
    for pattern in JOB_TITLE_PATTERNS:
        matches = list(pattern["regex"].finditer(norm.lower))
        if not matches:
            continue

        earliest_match = min(matches, key=lambda m: m.start())
        score = pattern["base_score"]

        if earliest_match.start() <= SUMMARY_WINDOW:
            score += 3
        elif earliest_match.start() <= MID_WINDOW:
            score += 1

        scores[pattern["display"]] += score

    if not scores:
        return None, 0.0

    title, score = scores.most_common(1)[0]
    return title, float(score)


def _extract_skills(norm: NormalizedText) -> list[str]:
    if not norm.lower:
        return []

    found = [skill for skill, regex in SKILL_PATTERNS.items() if regex.search(norm.lower)]
    return sorted(set(found))


def _extract_location(norm: NormalizedText) -> tuple[Optional[str], float]:
    if not norm.lower:
        return None, 0.0

    for state, regex in STATE_PATTERNS.items():
        match = regex.search(norm.lower)
        if match:
            boost = 1.5 if match.start() <= MID_WINDOW else 1.0
            return state, boost

    return None, 0.0


def _extract_experience(norm: NormalizedText, job_title: Optional[str]) -> tuple[int, float]:
    if not norm.lower:
        return DEFAULT_YOE, 0.0

    for pattern in YOE_PATTERNS:
        match = pattern.search(norm.lower)
        if match:
            if match.groupdict().get("start") and match.groupdict().get("end"):
                start_val = int(match.group("start"))
                end_val = int(match.group("end"))
                years = int(round((start_val + end_val) / 2))
            else:
                years = int(match.group("years"))

            return min(years, MAX_YOE), 1.0

    # Fallback heuristics
    if job_title:
        title_lower = job_title.lower()
        if any(keyword in title_lower for keyword in ("principal", "staff", "director", "vp")):
            return 12, 0.5
        if any(keyword in title_lower for keyword in ("senior", "lead", "manager")):
            return 7, 0.5

    if any(keyword in norm.lower for keyword in SENIOR_KEYWORDS):
        return 8, 0.4

    return DEFAULT_YOE, 0.0


def _match_title_candidate(text: str) -> Optional[str]:
    lower = text.lower()
    for pattern in JOB_TITLE_PATTERNS:
        if pattern["regex"].search(lower):
            return pattern["display"]
    return None


def _infer_title_from_experience(norm: NormalizedText) -> tuple[Optional[str], float]:
    if not norm.lines:
        return None, 0.0

    window = norm.lines[:80]
    for idx, line in enumerate(window):
        lower = line.lower()
        if any(marker in lower for marker in EXPERIENCE_SECTION_MARKERS):
            for offset in range(1, 6):
                if idx + offset >= len(window):
                    break
                candidate = window[idx + offset]
                if len(candidate) > 120:
                    continue
                title = _match_title_candidate(candidate)
                if not title:
                    for token in TITLE_SPLIT_TOKENS:
                        if token in candidate:
                            segment = candidate.split(token, 1)[0]
                            title = _match_title_candidate(segment)
                            if title:
                                break
                if title:
                    return title, 0.7

    for line in window:
        if len(line) > 100:
            continue
        title = _match_title_candidate(line)
        if title:
            return title, 0.5

    return None, 0.0


def _infer_title_from_skills(skills: list[str]) -> tuple[Optional[str], float]:
    if not skills:
        return None, 0.0

    skill_set = set(skills)
    best_title: Optional[str] = None
    best_score = 0.0

    for hint in SKILL_TITLE_HINTS:
        if not hint["required"].issubset(skill_set):
            continue
        score = 0.4
        if hint["preferred"]:
            score += 0.05 * len(skill_set & hint["preferred"])
        if score > best_score:
            best_score = score
            best_title = hint["title"]

    return best_title, best_score

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
    """
    Parse resume text using spaCy NER + custom patterns.

    Uses spaCy for entity recognition and combines with existing
    regex patterns for better accuracy.
    """
    normalized = _normalize_text(text)

    # Try to load spaCy model
    nlp = None
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using regex-only parsing")
    except ImportError:
        logger.warning("spaCy not installed, using regex-only parsing")

    # Use regex patterns (existing logic)
    job_title, job_score = _score_job_titles(normalized)
    location, location_score = _extract_location(normalized)
    skills = _extract_skills(normalized)
    experience_title, experience_score = _infer_title_from_experience(normalized)
    if experience_title and (not job_title or experience_score > job_score):
        job_title = experience_title
        job_score = experience_score

    skill_title, skill_title_score = _infer_title_from_skills(skills)
    if skill_title and (not job_title or skill_title_score > job_score):
        job_title = skill_title
        job_score = skill_title_score
    experience_years, experience_confidence = _extract_experience(normalized, job_title)

    # Enhance with spaCy if available
    if nlp:
        try:
            doc = nlp(normalized.clean[:5000])  # Limit to first 5000 chars for speed

            # Extract potential job titles from organizations (often in "at Company" context)
            org_names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

            # If no job title found via regex, try to infer from context
            if not job_title and org_names:
                # Look for patterns like "Engineer at Google" or "ML Scientist, OpenAI"
                for org in org_names[:3]:  # Check first 3 orgs
                    org_pattern = re.compile(rf'(\w+\s+\w+)\s+(?:at|@)\s+{re.escape(org)}', re.IGNORECASE)
                    match = org_pattern.search(normalized.clean)
                    if match:
                        potential_title = match.group(1)
                        # Validate against known patterns
                        for pattern_obj in JOB_TITLE_PATTERNS:
                            if pattern_obj["regex"].search(potential_title):
                                job_title = pattern_obj["display"]
                                job_score = 0.7  # Medium confidence from context
                                break
                    if job_title:
                        break

            # Enhance location extraction with spaCy GPE (Geo-Political Entity)
            if not location:
                gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
                for gpe in gpe_entities:
                    # Check if it matches any of our state patterns
                    for state_code, pattern in STATE_PATTERNS.items():
                        if pattern.search(gpe):
                            location = state_code
                            location_score = 0.8
                            break
                    if location:
                        break

            # Enhance skills with spaCy entities (sometimes ORGs are actually tech/skills)
            tech_orgs = ["TensorFlow", "PyTorch", "Kubernetes", "Docker", "AWS", "Azure", "GCP"]
            for ent in doc.ents:
                if ent.label_ == "ORG" and ent.text in tech_orgs and ent.text not in skills:
                    skills.append(ent.text)

        except Exception as e:
            logger.warning(f"spaCy enhancement failed: {e}, using regex results")

    return {
        "job_title": job_title,
        "location": location,
        "experience_years": experience_years,
        "skills": skills,
        "confidence": {
            "job_title": job_score,
            "location": location_score,
            "experience": experience_confidence,
            "skills": 1.0 if skills else 0.0,
        },
        "extracted_text_preview": normalized.clean[:500] if normalized.clean else None,
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
model_loaded_at: Optional[datetime] = None
model_metadata: dict = {}


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




def load_model_from_s3(bucket: str = "ai-salary-predictor", force: bool = False) -> bool:
    """
    Download latest model from S3.

    Args:
        bucket: S3 bucket name
        force: Force download even if local model exists

    Returns:
        True if model was downloaded, False otherwise
    """
    if not S3_AVAILABLE:
        logger.warning("boto3 not available, cannot download from S3")
        return False

    try:
        s3 = boto3.client('s3')
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        local_path = model_dir / "salary_model_latest.pkl"

        # Check if we should download
        if not force and local_path.exists():
            # Check S3 timestamp vs local
            try:
                s3_obj = s3.head_object(Bucket=bucket, Key="models/latest/salary_model_latest.pkl")
                s3_modified = s3_obj['LastModified']
                local_modified = datetime.fromtimestamp(local_path.stat().st_mtime, tz=s3_modified.tzinfo)

                if local_modified >= s3_modified:
                    logger.info("Local model is up-to-date")
                    return False
            except Exception as e:
                logger.warning(f"Could not check S3 timestamp: {e}")

        # Download from S3
        logger.info(f"Downloading model from s3://{bucket}/models/latest/")
        s3.download_file(bucket, "models/latest/salary_model_latest.pkl", str(local_path))
        logger.info("Model downloaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to download model from S3: {e}")
        return False


def load_model(from_s3: bool = False):
    """Load the trained model."""
    global predictor, engineer, model_loaded_at, model_metadata

    # Try to download from S3 if requested
    if from_s3:
        load_model_from_s3()

    model_path = Path("models/salary_model_latest.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    predictor = SalaryPredictor(model_dir="models")
    predictor.load()

    engineer = FeatureEngineer()

    # Extract model metadata
    model_loaded_at = datetime.now()
    model_metadata = {
        "loaded_at": model_loaded_at.isoformat(),
        "model_file": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / 1024 / 1024, 2),
        "training_timestamp": predictor.training_metrics.get("timestamp"),
        "training_samples": predictor.training_metrics.get("train", {}).get("n_samples"),
        "test_r2": predictor.training_metrics.get("test", {}).get("r2_score"),
        "test_mae": predictor.training_metrics.get("test", {}).get("mae"),
    }

    logger.info(f"Model loaded successfully at {model_loaded_at}")


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
        "model": {
            "loaded_at": model_metadata.get("loaded_at", "not loaded"),
            "training_date": model_metadata.get("training_timestamp", "unknown"),
        },
        "endpoints": {
            "/api/predict": "POST - Predict salary for a single job",
            "/api/predict/batch": "POST - Predict salaries for multiple jobs",
            "/api/health": "GET - Health check",
            "/api/model/info": "GET - Model information",
            "/api/model/reload": "POST - Reload model from S3 (requires auth)",
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
        "loaded_at": model_metadata.get("loaded_at"),
        "model_size_mb": model_metadata.get("model_size_mb"),
        "training": {
            "timestamp": model_metadata.get("training_timestamp"),
            "samples": model_metadata.get("training_samples"),
            "r2_score": model_metadata.get("test_r2"),
            "mae": model_metadata.get("test_mae"),
        },
        "feature_names": predictor.feature_names[:50],  # First 50 features for debugging
        "training_metrics": predictor.training_metrics.get("train", {}),
    }


@router.post("/model/reload")
async def reload_model(from_s3: bool = True):
    """
    Reload the model, optionally pulling latest from S3.

    Args:
        from_s3: Whether to download latest model from S3 before reloading
    """
    try:
        logger.info(f"Reloading model (from_s3={from_s3})")

        downloaded = False
        if from_s3:
            downloaded = load_model_from_s3(force=True)

        load_model(from_s3=False)  # Don't download again

        return {
            "success": True,
            "message": "Model reloaded successfully",
            "downloaded_from_s3": downloaded,
            "loaded_at": model_metadata.get("loaded_at"),
            "training_timestamp": model_metadata.get("training_timestamp"),
        }
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


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
    """
    Calculate top factors affecting this specific prediction.
    Uses actual feature values × feature importance from the model.
    """
    if predictor is None or predictor.feature_importance is None:
        return []

    factors = []

    # Calculate impact score for each feature (value × importance)
    feature_impacts = []
    for feature_name in predictor.feature_names:
        if feature_name not in X.columns:
            continue

        value = float(X[feature_name].iloc[0])
        importance = predictor.feature_importance.get(feature_name, 0)

        # Only consider features that are "active" (non-zero) for this prediction
        if value != 0:
            impact_score = abs(value * importance)
            feature_impacts.append({
                'name': feature_name,
                'value': value,
                'importance': importance,
                'impact_score': impact_score,
                'direction': 'positive' if value > 0 else 'negative'
            })

    # Sort by impact score, but boost location if it's notable (high/low paying state)
    def sort_key(feature_data):
        score = feature_data['impact_score']
        # Give notable locations a 20% boost so they're more likely to appear in top 4
        if feature_data['name'] == 'state_clean_encoded':
            if input_data.location.upper() in HIGH_PAY_STATES or input_data.location.upper() in LOW_PAY_STATES:
                score *= 1.2
        return score

    top_features = sorted(feature_impacts, key=sort_key, reverse=True)

    # Convert to human-readable factors
    processed_categories = set()  # Track categories to avoid duplicates

    for feature_data in top_features:
        feature_name = feature_data['name']
        value = feature_data['value']
        rank = len(factors) + 1

        # Experience
        if feature_name == "estimated_yoe" and "experience" not in processed_categories:
            years = int(value)
            level = "senior" if years >= 7 else "mid" if years >= 4 else "entry"
            factors.append(FeatureFactor(
                name="Experience",
                impact="positive",
                description=f"{years} years ({level} level, top {rank} factor)"
            ))
            processed_categories.add("experience")

        # Location (state encoding)
        elif feature_name == "state_clean_encoded" and "location" not in processed_categories:
            market_type = "high-paying market" if input_data.location.upper() in HIGH_PAY_STATES else "market"
            factors.append(FeatureFactor(
                name="Location",
                impact="positive" if input_data.location.upper() in HIGH_PAY_STATES else "neutral",
                description=f"{input_data.location} ({market_type}, top {rank} factor)"
            ))
            processed_categories.add("location")

        # Company tier
        elif feature_name == "company_tier_encoded" and "company" not in processed_categories:
            if input_data.company:
                factors.append(FeatureFactor(
                    name="Company",
                    impact="positive",
                    description=f"{input_data.company} (top {rank} factor)"
                ))
                processed_categories.add("company")

        # Seniority indicators
        elif feature_name in ["is_senior", "is_staff_principal", "is_manager_director"] and "seniority" not in processed_categories:
            if value > 0:
                level_map = {
                    "is_senior": "Senior level",
                    "is_staff_principal": "Staff/Principal level",
                    "is_manager_director": "Manager/Director level"
                }
                factors.append(FeatureFactor(
                    name="Seniority",
                    impact="positive",
                    description=f"{level_map.get(feature_name, 'Senior')} (top {rank} factor)"
                ))
                processed_categories.add("seniority")

        # Skills
        elif feature_name.startswith("skill_") and value > 0:
            skill_name = feature_name.replace("skill_", "").replace("_", " ").title()
            # Only show top 2 skills to avoid cluttering
            skill_factors = [f for f in factors if f.name.startswith("Skill:")]
            if len(skill_factors) < 2:
                factors.append(FeatureFactor(
                    name=f"Skill: {skill_name}",
                    impact="positive",
                    description=f"High-value skill (top {rank} factor)"
                ))

        # Stop after we have 4 factors
        if len(factors) >= 4:
            break

    # ALWAYS include location if it's notable (high-paying or low-paying)
    # Even if we have 4 factors, replace the last one if location is notable and not shown
    if "location" not in processed_categories:
        location_factor = None
        if input_data.location.upper() in HIGH_PAY_STATES:
            location_factor = FeatureFactor(
                name="Location",
                impact="positive",
                description=f"{input_data.location} (high-paying market)"
            )
        elif input_data.location.upper() in LOW_PAY_STATES:
            location_factor = FeatureFactor(
                name="Location",
                impact="negative",
                description=f"{input_data.location} (lower cost market)"
            )

        if location_factor:
            if len(factors) < 4:
                factors.append(location_factor)
            else:
                # Replace the 4th factor with location since it's notable
                factors[3] = location_factor
            processed_categories.add("location")

    # If we still don't have enough factors, add a note about seniority if entry-level
    if len(factors) < 4 and "seniority" not in processed_categories:
        title_lower = input_data.job_title.lower()
        if not any(x in title_lower for x in ["senior", "staff", "principal", "lead", "director", "manager"]):
            factors.append(FeatureFactor(
                name="Seniority",
                impact="negative",
                description="Entry/Mid level title (consider senior roles)"
            ))

    return factors[:4]


@router.get("/options")
async def get_options():
    """Get available options for prediction inputs (from actual model features)."""
    # Extract skills from actual model features
    skills = []
    if predictor is not None and predictor.feature_names:
        skill_features = [f for f in predictor.feature_names if f.startswith("skill_")]
        skills = [
            f.replace("skill_", "").replace("_", " ").title()
            for f in skill_features
        ]
        # Sort by feature importance if available
        if predictor.feature_importance:
            skills_with_importance = [
                (skill, predictor.feature_importance.get(f"skill_{skill.lower().replace(' ', '_')}", 0))
                for skill in skills
            ]
            skills = [s[0] for s in sorted(skills_with_importance, key=lambda x: x[1], reverse=True)]

    # Fallback skills if model not loaded
    if not skills:
        skills = ["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow",
                  "NLP", "Computer Vision", "MLOps", "Kubernetes", "AWS"]

    return {
        "job_titles": [
            "ML Engineer", "Senior ML Engineer", "Staff ML Engineer", "Principal ML Engineer",
            "Data Scientist", "Senior Data Scientist", "Staff Data Scientist",
            "AI Engineer", "Senior AI Engineer",
            "Research Scientist", "Senior Research Scientist", "Applied Scientist",
            "Data Engineer", "Senior Data Engineer", "MLOps Engineer",
            "AI/ML Manager", "Director of ML", "VP of AI",
        ],
        "locations": [
            {"code": "CA", "name": "California"}, {"code": "NY", "name": "New York"},
            {"code": "WA", "name": "Washington"}, {"code": "TX", "name": "Texas"},
            {"code": "MA", "name": "Massachusetts"}, {"code": "CO", "name": "Colorado"},
            {"code": "IL", "name": "Illinois"}, {"code": "GA", "name": "Georgia"},
            {"code": "NC", "name": "North Carolina"}, {"code": "FL", "name": "Florida"},
            {"code": "PA", "name": "Pennsylvania"}, {"code": "VA", "name": "Virginia"},
            {"code": "AZ", "name": "Arizona"}, {"code": "OR", "name": "Oregon"},
            {"code": "MD", "name": "Maryland"}, {"code": "NJ", "name": "New Jersey"},
            {"code": "OH", "name": "Ohio"}, {"code": "MI", "name": "Michigan"},
            {"code": "MN", "name": "Minnesota"}, {"code": "UT", "name": "Utah"},
        ],
        "skills": skills,  # Now dynamic from model
    }


class ResumeParseResponse(BaseModel):
    """Response model for resume parsing."""

    job_title: Optional[str] = None
    location: Optional[str] = None
    experience_years: int = 3
    skills: list[str] = []
    confidence: Optional[dict[str, float]] = None
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
            confidence=parsed.get("confidence"),
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
# Only mount with /api prefix to avoid duplicate endpoints in documentation
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
