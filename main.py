import uvicorn
from fastapi import FastAPI
from app.routes import router as chat_router
from fastapi.middleware.cors import CORSMiddleware
from app.api.validator.routes import router as validator_router 

# --------------------------------------------
# FastAPI App Initialization
# --------------------------------------------
app = FastAPI(
    title="Agent Streaming API",
    version="1.0.0",
    description="An API for interacting with AI agents, supporting both streaming and normal responses."
)

# --------------------------------------------
# CORS Configuration
# --------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],        # local dev exact match
    allow_origin_regex=r"https?://.*\.hiptraveler\.com|https?://hiptraveler\.com",  # all subdomains + root
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------
# Routes
# --------------------------------------------

    
@app.get("/")
async def root():
    """Root endpoint providing a welcome message."""
    return {"message": "Agent Streaming API is running. Go to /docs for API documentation."}

@app.get("/health")
def health():
    """Health check endpoint for ALB and ECS."""
    return {"ok": True}

# Include the chat router
app.include_router(chat_router)

# Mount validator app under /validator
app.include_router(validator_router, prefix="/validator", tags=["validator"])

# --------------------------------------------
# Uvicorn Entrypoint
# --------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
