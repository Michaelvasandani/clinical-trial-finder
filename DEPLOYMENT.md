# Clinical Trial Finder - Deployment Guide

This guide covers deploying the Clinical Trial Finder application to cloud platforms like Render and Vercel.

## Overview

The application consists of two main components:
- **FastAPI Backend** (`chat_api.py`) - Handles AI processing, database operations, and API endpoints
- **Streamlit Frontend** (`streamlit_app.py`) - Provides the user interface

## Deployment Options

### Option 1: Full Render Deployment (Recommended)

Deploy both frontend and backend on Render with managed PostgreSQL.

#### Services Required:
1. **PostgreSQL Database** - Managed database service
2. **FastAPI Backend** - Web service
3. **Streamlit Frontend** - Web service

#### Step-by-Step Render Deployment:

1. **Create Render Account**
   - Sign up at [render.com](https://render.com)
   - Connect your GitHub repository

2. **Deploy PostgreSQL Database**
   - Create new PostgreSQL service
   - Name: `clinical-trial-finder-db`
   - Plan: Starter ($7/month)
   - Note the connection string

3. **Deploy FastAPI Backend**
   - Create new Web Service
   - Name: `clinical-trial-finder-api`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python chat_api.py`
   - Plan: Starter ($7/month)

   **Environment Variables:**
   ```
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=postgresql://user:pass@host:port/dbname  # Auto-provided by Render
   ENVIRONMENT=production
   DATA_PATH=/opt/render/project/data
   EMBEDDINGS_PATH=/opt/render/project/embeddings
   CONVERSATIONS_PATH=/opt/render/project/conversations
   LOGS_PATH=/opt/render/project/logs
   INPUT_CSV_NAME=clinical_trials_latest.csv
   CORS_ORIGINS=https://your-frontend-url.onrender.com
   ```

4. **Deploy Streamlit Frontend**
   - Create new Web Service
   - Name: `clinical-trial-finder-ui`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
   - Plan: Starter ($7/month)

   **Environment Variables:**
   ```
   API_BASE_URL=https://your-backend-url.onrender.com
   ```

5. **Upload Data and Embeddings**
   - Use Render's persistent disk feature
   - Upload your `clinical_trials_*.csv` file as `clinical_trials_latest.csv`
   - Upload embeddings from your local `/embeddings/` directory

### Option 2: Hybrid Deployment (Vercel + Render)

Deploy Streamlit frontend on Vercel and FastAPI backend on Render.

#### Vercel Frontend Deployment:

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Deploy to Vercel**
   ```bash
   vercel --prod
   ```

3. **Configure Environment Variables in Vercel**
   ```
   API_BASE_URL=https://your-render-backend.onrender.com
   ```

#### Render Backend Deployment:
Follow steps 2-3 from Option 1 above.

## Configuration Files

The repository includes deployment configuration files:

- `render.yaml` - Render infrastructure as code
- `Dockerfile` - Container configuration
- `vercel.json` - Vercel deployment settings

## Environment Variables Reference

### Required for All Deployments:
```bash
OPENAI_API_KEY=sk-...                    # Your OpenAI API key
DATABASE_URL=postgresql://...            # PostgreSQL connection string
ENVIRONMENT=production                   # Deployment environment
```

### Optional Configuration:
```bash
# Paths (defaults provided for Render)
DATA_PATH=/opt/render/project/data
EMBEDDINGS_PATH=/opt/render/project/embeddings
CONVERSATIONS_PATH=/opt/render/project/conversations
LOGS_PATH=/opt/render/project/logs

# Data Configuration
INPUT_CSV_NAME=clinical_trials_latest.csv

# API Configuration
CORS_ORIGINS=https://your-frontend.com
API_BASE_URL=https://your-backend.com

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=50
MAX_REQUESTS_PER_HOUR=1000
MAX_DAILY_COST_USD=50.0
```

## Data Migration

### Initial Data Setup:

1. **Upload Clinical Trial Data**
   - Copy your latest `clinical_trials_*.csv` file
   - Rename to `clinical_trials_latest.csv`
   - Upload to production data directory

2. **Upload Embeddings**
   - Copy entire `/embeddings/` directory to production
   - Ensure files: `clinical_trials.index`, `clinical_trials_metadata.json`, `chunk_mapping.json`

3. **Initialize Database**
   - The vector database will be automatically initialized on first run
   - Embeddings will be loaded from the uploaded files

### Data Update Process:

1. **Update Trial Data**
   - Run `python main.py` locally with new data
   - Upload new CSV file as `clinical_trials_latest.csv`

2. **Update Embeddings**
   - Run `python generate_embeddings.py` locally
   - Upload new embedding files to production

3. **Restart Services**
   - Restart both frontend and backend services to pick up new data

## Monitoring and Maintenance

### Health Checks:
- API Health: `GET https://your-api.onrender.com/health`
- System Stats: `GET https://your-api.onrender.com/stats`

### Logging:
- Render provides built-in logging for all services
- Conversation logs are stored in `/conversations/` directory
- Error logs available in Render dashboard

### Performance:
- Monitor response times via Render dashboard
- Database performance via PostgreSQL metrics
- Consider upgrading plans for higher traffic

## Cost Estimation

### Render Full Deployment:
- PostgreSQL: $7/month
- FastAPI Backend: $7/month
- Streamlit Frontend: $7/month
- **Total: ~$21/month**

### Hybrid (Vercel + Render):
- Render Backend + DB: $14/month
- Vercel Frontend: Free tier
- **Total: ~$14/month**

## Troubleshooting

### Common Issues:

1. **Database Connection Errors**
   - Verify DATABASE_URL environment variable
   - Check PostgreSQL service status
   - Ensure database allows external connections

2. **Missing Embeddings**
   - Verify embeddings files uploaded correctly
   - Check file paths in environment variables
   - Restart backend service

3. **API Connection Errors**
   - Verify API_BASE_URL in frontend
   - Check CORS_ORIGINS in backend
   - Ensure both services are running

4. **Performance Issues**
   - Monitor memory usage (embeddings are memory-intensive)
   - Consider upgrading to higher plans
   - Optimize embedding file sizes

### Support Resources:
- Render Documentation: [render.com/docs](https://render.com/docs)
- Vercel Documentation: [vercel.com/docs](https://vercel.com/docs)
- OpenAI API Status: [status.openai.com](https://status.openai.com)

## Security Considerations

1. **API Keys**
   - Store OpenAI API key securely in environment variables
   - Rotate keys regularly
   - Monitor API usage and costs

2. **Database Security**
   - Use strong passwords
   - Enable SSL connections
   - Regular backups

3. **CORS Configuration**
   - Restrict CORS origins to your frontend domains
   - Avoid wildcard (*) origins in production

4. **Rate Limiting**
   - Configure appropriate rate limits
   - Monitor for abuse
   - Set cost alerts

## Next Steps

After successful deployment:

1. **Test All Features**
   - Patient extraction and matching
   - Conversation functionality
   - Trial search and explanations

2. **Set Up Monitoring**
   - Configure alerts for errors
   - Monitor API usage and costs
   - Set up uptime monitoring

3. **Regular Maintenance**
   - Update trial data monthly
   - Monitor and optimize performance
   - Keep dependencies updated

For additional support or questions, refer to the main README.md file or create an issue in the repository.