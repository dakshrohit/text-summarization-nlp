from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from textSummarizer.pipeline.prediction import PredictionPipeline
from textSummarizer.logging import logger

# Load environment variables
load_dotenv()

# Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
MAX_TEXT_LENGTH = 1024
MIN_TEXT_LENGTH = 10

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model_trainer/pegasus-samsum-model")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "artifacts/model_trainer/tokenizer")

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Artificial intelligence is transforming industries by automating tasks and improving decision-making. Machine learning models can process vast amounts of data quickly and efficiently."
            }
        }

class PredictionResponse(BaseModel):
    summary: str

class TrainingResponse(BaseModel):
    status: str
    message: str

# Create FastAPI app
app = FastAPI(
    title="Text Summarization API",
    description="End-to-end Text Summarization using Fine-tuned PEGASUS model",
    version="1.0.0"
)

# Global prediction pipeline
prediction_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize prediction pipeline on startup"""
    global prediction_pipeline
    try:
        logger.info("Initializing prediction pipeline...")
        
        # Check if model files exist
        if not Path(MODEL_PATH).exists():
            logger.warning(f"Model not found at {MODEL_PATH}. Run /train endpoint first.")
        if not Path(TOKENIZER_PATH).exists():
            logger.warning(f"Tokenizer not found at {TOKENIZER_PATH}. Run /train endpoint first.")
        
        prediction_pipeline = PredictionPipeline()
        logger.info("Prediction pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prediction pipeline: {str(e)}")
        # Continue even if initialization fails - will error at prediction time

@app.get("/", tags=["root"])
async def index():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "message": "API is running"}
    )

@app.get("/train", tags=["training"])
async def training():
    """Train the model from scratch. Warning: This is a long-running operation."""
    try:
        logger.info("Starting model training...")
        
        # Use subprocess instead of os.system for better security and control
        result = subprocess.run(
            ["python", "main.py"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("Training completed successfully")
            return TrainingResponse(
                status="success",
                message="Training completed successfully! Model and tokenizer saved."
            )
        else:
            error_msg = result.stderr or "Unknown error occurred"
            logger.error(f"Training failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {error_msg}"
            )
            
    except subprocess.TimeoutExpired:
        logger.error("Training timeout after 1 hour")
        raise HTTPException(
            status_code=504,
            detail="Training timeout - operation exceeded 1 hour limit"
        )
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_route(request: PredictionRequest):
    """
    Summarize the provided text.
    
    - **text**: The input text to summarize (between {MIN_TEXT_LENGTH} and {MAX_TEXT_LENGTH} characters)
    """
    try:
        # Validate input
        if not request.text:
            raise HTTPException(
                status_code=400,
                detail="Text field cannot be empty"
            )
        
        # Clean and validate text length
        text = request.text.strip()
        
        if len(text) < MIN_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text too short. Minimum length is {MIN_TEXT_LENGTH} characters"
            )
        
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Text too long. Maximum length is {MAX_TEXT_LENGTH} characters"
            )
        
        # Check if model is available
        if prediction_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction service not available. Model not initialized. Please run /train first."
            )
        
        # Check if model files exist
        if not Path(MODEL_PATH).exists() or not Path(TOKENIZER_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model files not found. Please ensure training has completed. "
                        f"Run the /train endpoint to train the model."
            )
        
        logger.info(f"Processing prediction request (text length: {len(text)})")
        
        # Generate summary
        summary = prediction_pipeline.predict(text)
        
        if not summary:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate summary"
            )
        
        logger.info(f"Prediction successful (summary length: {len(summary)})")
        
        return PredictionResponse(summary=summary)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    logger.info(f"Starting FastAPI server on {HOST}:{PORT} (Debug: {DEBUG})")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        debug=DEBUG,
        log_level="info"
    )
