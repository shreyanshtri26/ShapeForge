import os
import logging
import time
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import psutil
import uvicorn
from dotenv import load_dotenv
from pathlib import Path
from model import LocalLLM, get_llm_instance

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "llm_service.log")),
    ],
)
logger = logging.getLogger("llm_service")

# Try to load .env file from project root
env_path = Path(__file__).parents[2] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from {env_path}")


# Initialize FastAPI app
app = FastAPI(title="Local LLM Service", description="API for local LLM interaction")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request started: {request.method} {request.url.path}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"Request completed: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.4f}s"
    )

    return response


# Model request and response classes
class PromptRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ExpandRequest(BaseModel):
    prompt: str


class LLMResponse(BaseModel):
    text: str


# Global LLM instance
llm = None


@app.on_event("startup")
async def startup_event():
    """Initialize the LLM on startup"""
    global llm
    logger.info("Starting LLM service initialization...")

    # Determine model path from environment or local directory
    local_models_dir = Path(__file__).parent / "models"
    model_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
    model_path = os.environ.get("MODEL_PATH")

    # Check for locally downloaded model first
    local_model_path = local_models_dir / model_id
    if local_model_path.exists():
        logger.info(f"Found local model at: {local_model_path}")
        model_path = str(local_model_path)
    elif model_path and os.path.isdir(model_path):
        logger.info(f"Using local model from MODEL_PATH: {model_path}")
    else:
        # Fall back to downloading from HuggingFace
        logger.info(f"Using model ID from Hugging Face: {model_id}")
        model_path = model_id

        # Warning for gated models without token
        if "meta-llama" in model_id and not os.environ.get("HF_TOKEN"):
            logger.warning(
                f"Using Meta-Llama model without HF_TOKEN. Authentication will likely fail."
            )

    # List of fallback models in order of preference
    fallback_models = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2b-it",
    ]

    # Try loading the primary model first, then fallbacks
    start_time = time.time()
    try:
        llm = get_llm_instance(model_path)
        init_time = time.time() - start_time
        logger.info(
            f"LLM initialized successfully with model: {model_path} in {init_time:.2f} seconds"
        )

        # Log memory usage if psutil is available
        try:
            memory = psutil.virtual_memory()
            logger.info(
                f"System memory: {memory.percent}% used ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)"
            )
        except (ImportError, NameError):
            pass

    except Exception as e:
        logger.error(f"Failed to initialize primary model: {str(e)}")

        # Try each fallback model in sequence
        for i, fallback in enumerate(fallback_models):
            try:
                logger.info(f"Attempting to load fallback model #{i+1}: {fallback}")
                llm = get_llm_instance(fallback)
                logger.info(f"Successfully loaded fallback model: {fallback}")
                break
            except Exception as fallback_error:
                logger.error(f"Fallback model {fallback} failed: {str(fallback_error)}")

        if not llm:
            logger.error("All models failed to load. Service will respond with errors.")


@app.post("/generate", response_model=LLMResponse)
async def generate_text(request: PromptRequest):
    """Generate text based on a prompt"""
    logger.info(
        f"Received text generation request, prompt length: {len(request.prompt)} chars"
    )
    logger.debug(f"Prompt: {request.prompt[:50]}...")

    if not llm:
        logger.error("LLM service not initialized when generate endpoint was called")
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    try:
        start_time = time.time()

        logger.info(
            f"Generation parameters: max_tokens={request.max_tokens}, temperature={request.temperature}, top_p={request.top_p}"
        )

        response = llm.generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        generation_time = time.time() - start_time
        response_length = len(response)

        logger.info(
            f"Text generation completed in {generation_time:.2f}s, response length: {response_length} chars"
        )
        logger.debug(f"Generated response: {response[:50]}...")

        return LLMResponse(text=response)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/expand", response_model=LLMResponse)
async def expand_prompt(request: ExpandRequest):
    """Expand a creative prompt with rich details"""
    logger.info(f"Received prompt expansion request, prompt: '{request.prompt}'")

    if not llm:
        logger.error("LLM service not initialized when expand endpoint was called")
        raise HTTPException(status_code=503, detail="LLM service not initialized")

    try:
        start_time = time.time()

        expanded = llm.expand_creative_prompt(request.prompt)

        expansion_time = time.time() - start_time
        expanded_length = len(expanded)

        logger.info(
            f"Prompt expansion completed in {expansion_time:.2f}s, original length: {len(request.prompt)}, expanded length: {expanded_length}"
        )
        logger.debug(f"Original: '{request.prompt}'")
        logger.debug(f"Expanded: '{expanded}'")

        return LLMResponse(text=expanded)
    except Exception as e:
        logger.error(f"Error expanding prompt: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check endpoint called")

    if llm:
        logger.info(f"Health check: LLM service is healthy, model: {llm.model_path}")
        return {"status": "healthy", "model": llm.model_path}

    logger.warning("Health check: LLM service is still initializing")
    return {"status": "initializing"}


# Start the service if run directly
if __name__ == "__main__":

    # Check for psutil dependency
    try:
        import psutil
    except ImportError:
        logger.warning(
            "psutil not installed. Some system resource metrics will not be available."
        )
        logger.warning("Install with: pip install psutil")

    logger.info("Starting LLM service server")
    uvicorn.run(app, host="0.0.0.0", port=8001)
