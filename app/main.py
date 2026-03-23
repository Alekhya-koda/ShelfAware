import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine, Base

# Import models so SQLAlchemy registers tables/relationships
from app.models import user, book, genre, book_genre, bookshelf, synopsis_moderation  # noqa: F401
try:
    from app.services.synopsis_scheduler import SynopsisScheduler
except ImportError:
    SynopsisScheduler = None

# Import routers (ROUTES, not models)
from app.routes import auth, books, chatbot
from app.routes.admin import router as admin_router
from app.routes.bookshelf import router as bookshelf_router
from app.routes import chroma  # ChromaDB search routes
from app.routes import user_profile
from app.routes import review
from app.routes import recommendation_routes


# This helper class is for serving the Single Page Application (SPA).
# It falls back to serving 'index.html' for any path that is not found,
# which allows client-side routing to work correctly.
class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except HTTPException as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            raise ex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create tables on startup
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI application."""
    # Startup: Initialize and start synopsis scheduler
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if SynopsisScheduler is None:
            logger.warning("Synopsis scheduler module not available. Synopsis sync disabled.")
        elif openai_api_key:
            # Initialize scheduler
            SynopsisScheduler.initialize(openai_api_key=openai_api_key)
            # Start scheduler (runs daily at midnight UTC by default)
            SynopsisScheduler.start(hour=0, minute=0)
            logger.info("Synopsis scheduler started successfully")
        else:
            logger.warning("OPENAI_API_KEY environment variable not set. Synopsis sync disabled.")
    except Exception as e:
        logger.error(f"Failed to start synopsis scheduler: {str(e)}")

    yield

    # Shutdown: Stop the scheduler
    if SynopsisScheduler is not None:
        SynopsisScheduler.stop()
        logger.info("Synopsis scheduler stopped")


app = FastAPI(
    title="ShelfAware API",
    description="An API for managing books and integrating with Ollama and ChromaDB",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS to allow frontend requests
# Use environment variable CORS_ORIGINS to customize on deploy (comma-separated values)
# Default includes local dev and both static host ports (80 + 8000).
cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:4173,http://localhost:4176,http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:5176,http://localhost:3000,http://54.179.230.205,http://54.179.230.205:80,http://54.179.230.205:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])
app.include_router(books.router, prefix="/books", tags=["Books"])
app.include_router(bookshelf_router, prefix="/bookshelf", tags=["Bookshelf"])
app.include_router(chroma.router, prefix="/books/search", tags=["Books Search"])
app.include_router(user_profile.router, prefix="/user-profile", tags=["User Profile"])
app.include_router(review.router, prefix="/reviews", tags=["Reviews"])
app.include_router(
    recommendation_routes.router,
    prefix="/api",
    tags=["Recommendations"],
)
app.include_router(chatbot.router, prefix="/api/chatbot", tags=["Chatbot"])

logger.info(f"DEBUG: App routes after inclusion: {[r.path for r in app.routes]}") # TEMPORARY DEBUG LINE

@app.post("/admin/trigger-scheduler-sync")
def trigger_manual_sync():
    """
    Manual endpoint to trigger synopsis synchronization.
    Useful for testing and on-demand updates.

    Requires admin access in production.
    """
    try:
        if SynopsisScheduler is None:
            return {"status": "error", "message": "Synopsis scheduler module not available."}
        result = SynopsisScheduler.add_manual_job()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Manual sync failed: {str(e)}")
        return {"status": "error", "message": str(e)}


# Mount the static files directory to serve the frontend.
# This must be mounted AFTER all API routes are registered.
app.mount("/", SPAStaticFiles(directory="app/static"), name="static-app")