# ShelfAware - Community-Driven Book Synopsis System

ShelfAware is a FastAPI-based application for managing book bookshelves with intelligent community synopsis generation.

## Features

- **User-Generated Synopses**: Users can add personal synopses for books they're reading
- **Automated Synopsis Aggregation**: Daily cron job that collects user synopses and generates comprehensive community synopses using OpenAI LLM
- **Smart Comparison**: Intelligently compares existing community synopses with new user input
- **Book Management**: Track books with metadata including authors, genres, and publication info
- **Bookshelf Management**: Maintain personal bookshelves with reading status tracking

## Project Structure

```
ShelfAware/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── db/
│   │   ├── __init__.py
│   │   └── database.py         # SQLAlchemy configuration
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── author.py
│   │   ├── book.py
│   │   ├── bookshelf.py
│   │   ├── genre.py
│   │   ├── mood.py
│   │   ├── review.py
│   │   └── user.py
│   ├── routes/                 # API endpoints
│   │   └── books.py
│   ├── services/               # Business logic
│   │   ├── book_service.py
│   │   ├── review_service.py
│   │   ├── synopsis_sync_service.py      # Synopsis synchronization logic
│   │   └── (scheduler removed – sync now manual via `/admin/sync-synopses`)
│   └── dependencies/           # Dependency injection & utilities
│       ├── auth.py
│       ├── db.py
│       └── services.py
├── migrations/                 # Alembic database migrations
├── requirements.txt            # Python dependencies
├── run_tests.py               # Main test runner (unit & integration)
├── tests/                      # Unit and integration tests
│   ├── test_book_crud.py
│   ├── test_book_routes.py
│   ├── test_recommendation_engine.py
│   ├── run_book_tests.py        # convenience script for book CRUD tests
│   └── README.md                # testing documentation
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd ShelfAware
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 5. Set up database
```bash
# Run migrations if using Alembic
alembic upgrade head
```

### 6. Run the application
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## Synopsis Synchronization Cron Job

### How It Works

The system includes an automated daily cron job that:

1. **Extracts** all user-generated synopses from the bookshelf table
2. **Groups** synopses by book
3. **Compares** them with existing community synopses
4. **Generates** new community synopses using OpenAI's GPT-3.5-turbo model
5. **Updates** the database with improved synopses

### Configuration

#### Set Scheduler Time

Edit `app/main.py` startup event to customize when the job runs (UTC timezone):

```python
# Default: runs daily at midnight UTC
SynopsisScheduler.start(hour=0, minute=0)

# Example: run at 2:30 AM UTC
SynopsisScheduler.start(hour=2, minute=30)

# Example: run at 6 PM UTC
SynopsisScheduler.start(hour=18, minute=0)
```

#### Environment Variables

Set your OpenAI API key in `.env`:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

### API Usage

#### Manual Sync Trigger

To manually trigger a synopsis sync (useful for testing):

```bash
curl -X POST http://localhost:8000/admin/sync-synopses
```

Response:
```json
{
  "status": "success",
  "data": {
    "status": "completed",
    "timestamp": "2024-02-10T12:00:00.000000",
    "total_books_processed": 15,
    "updated": 8,
    "skipped": 7,
    "errors": []
  }
}
```

### Database Changes

Make sure your database includes the `CommunitySynopsis` column in the `book` table:

```sql
ALTER TABLE book ADD COLUMN CommunitySynopsis VARCHAR;
```

Or create a new migration:

```bash
alembic revision --autogenerate -m "Add CommunitySynopsis to Book model"
alembic upgrade head
```

### Service Classes

#### SynopsisSyncService

Located in `app/services/synopsis_sync_service.py`

**Methods:**
- `get_all_user_synopses(db, book_id=None)`: Retrieve user synopses from database
- `generate_community_synopsis(title, user_synopses)`: Generate synopsis using OpenAI
- `compare_synopses(current_synopsis, user_synopses)`: Determine if update is needed
- `sync_all_synopses(db)`: Main synchronization method

**Example Usage:**
```python
from app.services.synopsis_sync_service import SynopsisSyncService
from app.db.database import SessionLocal

service = SynopsisSyncService(openai_api_key="sk-...")
db = SessionLocal()

result = service.sync_all_synopses(db)
print(result)  # {'status': 'completed', 'updated': X, 'skipped': Y, ...}
```

#### SynopsisScheduler

Located in `app/services/synopsis_scheduler.py`

**Methods:**
- `initialize(openai_api_key)`: Initialize the scheduler
- `start(hour=0, minute=0)`: Start cron job with specified time
- `stop()`: Stop the scheduler
- `add_manual_job(book_id=None)`: Manually trigger sync
- `get_scheduler()`: Get scheduler instance

**Example Usage:**
```python
from app.services.synopsis_scheduler import SynopsisScheduler

# Initialize
SynopsisScheduler.initialize(openai_api_key="sk-...")

# Start daily at 3 AM UTC
SynopsisScheduler.start(hour=3, minute=0)

# Manually trigger
result = SynopsisScheduler.add_manual_job()

# Stop
SynopsisScheduler.stop()
```

## Data Models

### Book Model

```python
class Book(Base):
    __tablename__ = "book"
    
    book_id: str          # Primary key (UUID)
    title: str            # Book title
    subtitle: str (optional)
    cover_image_url: str (optional)
    abstract: str (optional)
    CommunitySynopsis: str (optional)  # Generated community synopsis
    page_count: int (optional)
    published_date: date (optional)
    created_at: datetime
    bookshelves: relationship    # Many-to-many with users
```

### Bookshelf Model

```python
class Bookshelf(Base):
    __tablename__ = "bookshelf"
    
    user_id: str          # Foreign key to user
    book_id: str          # Foreign key to book (composite primary key)
    shelf_status: str     # want_to_read, reading, completed
    date_added: datetime
    date_started: datetime (optional)
    date_finished: datetime (optional)
    updated_at: datetime
    Synopsis: str (optional)   # User-generated synopsis
```

## Dependencies

- **fastapi**: Web framework
- **sqlalchemy**: ORM
- **pydantic**: Data validation
- **openai**: GPT API integration
- **apscheduler**: Job scheduling
- **alembic**: Database migrations
- **python-jose**: JWT authentication
- **boto3**: AWS integration (optional)
- **langchain-openai**: LLM operations
- **chromadb**: Vector database

## Logging

The application logs all important events including:
- Scheduler startup/shutdown
- Cron job execution
- Synopsis generation
- Database updates
- Error details

View logs in your application output or configure log files as needed.

## Error Handling

The synopsis sync service handles:
- Missing OpenAI API key
- Database connection errors
- Invalid user synopses
- LLM generation failures
- Partial sync failures (continues processing other books)

All errors are logged with full context for debugging.

## Performance Notes

- Each synopsis generation costs money with OpenAI (uses GPT-3.5-turbo)
- Daily runs help keep synopses fresh without excessive API calls
- Processing time depends on number of books and user synopses
- Database queries are optimized with filters

## Future Enhancements

- [ ] Batch OpenAI requests for cost optimization
- [ ] Caching of similar synopses
- [ ] Sentiment analysis of user synopses
- [ ] Support for multiple languages
- [ ] Custom synopsis prompt templates
- [ ] Rate limiting for API calls
- [ ] Dashboard for sync job monitoring

## Production Deployment

### Quick Start (AWS ECS)

For detailed AWS ECS deployment instructions, see [`ECS_DEPLOYMENT_README.md`](ECS_DEPLOYMENT_README.md).

#### Prerequisites
- AWS Account with ECR, ECS, and RDS access
- Docker installed locally
- AWS CLI configured

#### Deploy to ECS
```bash
# 1. Update configuration files with your AWS account ID
# Edit deploy-to-ecs.sh and task-definitions/*.json

# 2. Create ECR repositories
aws ecr create-repository --repository-name shelfaware-backend --region ap-southeast-1
aws ecr create-repository --repository-name shelfaware-frontend --region ap-southeast-1

# 3. Deploy images
chmod +x deploy-to-ecs.sh
./deploy-to-ecs.sh

# 4. Create ECS cluster and services
# Follow the detailed steps in ECS_DEPLOYMENT_README.md
```

### Manual Backend Deployment

1. **Environment Setup**:
   ```bash
   cp .env.example .env
   # Edit .env with your production values
   ```

2. **Required Environment Variables**:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DATABASE_URL=postgresql://user:password@your-rds-endpoint:5432/shelfaware
   COGNITO_REGION=your_cognito_region
   COGNITO_USER_POOL_ID=your_user_pool_id
   COGNITO_CLIENT_ID=your_client_id
   COGNITO_CLIENT_SECRET=your_client_secret
   ```

3. **Database**:
   - For production, use PostgreSQL (AWS RDS recommended)
   - Update `DATABASE_URL` in `.env`

4. **CORS Configuration**:
   - Update `app/main.py` to include your production frontend domain in `allow_origins`
   - Example: `"https://your-frontend-domain.com"`

5. **Build and Run**:
   ```bash
   # Using Docker
   docker build -t shelfaware-backend .
   docker run -p 8000:8000 --env-file .env shelfaware-backend

   # Or using uvicorn directly
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Manual Frontend Deployment

1. **Environment Setup**:
   ```bash
   cd ref_frontend
   cp .env.example .env  # If exists, or create .env
   ```

2. **Update API URL**:
   ```env
   VITE_API_BASE_URL=https://your-backend-domain.com
   ```

3. **Build and Deploy**:
   ```bash
   npm install
   npm run build

   # Using Docker
   docker build -t shelfaware-frontend .
   docker run -p 80:80 shelfaware-frontend
   ```

### Infrastructure Recommendations

- **Backend**: Deploy to AWS ECS, Google Cloud Run, or similar container service
- **Frontend**: Deploy to AWS S3 + CloudFront, Vercel, or Netlify
- **Database**: Use AWS RDS (PostgreSQL) or Google Cloud SQL for production
- **Authentication**: Ensure Cognito is properly configured with correct user pools

### Security Considerations

- Use HTTPS in production
- Store secrets securely (AWS Secrets Manager, etc.)
- Configure proper CORS policies
- Set up monitoring and logging
- Regular security updates for dependencies

## Troubleshooting

### Scheduler not starting
- Check if `OPENAI_API_KEY` is set in `.env`
- Review application logs for initialization errors
- Ensure APScheduler is installed: `pip install apscheduler`

### Synopses not being updated
- Check if user synopses exist in database (non-null `Synopsis` field)
- Monitor OpenAI API quota and billing
- Verify database connectivity
- Check logs for specific error messages

### High API costs
- Reduce cron job frequency
- Increase the comparison threshold (modify `compare_synopses` method)
- Use cheaper OpenAI models (currently using GPT-3.5-turbo)

## License

[Your License Here]

## Support

For issues or questions, please create an issue in the repository.
