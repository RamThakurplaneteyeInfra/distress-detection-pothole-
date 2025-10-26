# Deployment Guide for Pothole Detection App

## Platform Comparison for Free Deployment

### 1. **Google Cloud Run** (Recommended)

**Free Tier**: 2M requests/month, 400K GB-seconds
**Resources**: Up to 2GB RAM, 8GB storage
**Pros**:

- Generous free tier
- Good for ML workloads
- Auto-scaling
- Easy PostgreSQL integration

**Deployment Steps**:

```bash
# Install Google Cloud CLI
# Create project and enable APIs
gcloud projects create your-project-id
gcloud config set project your-project-id

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Build and deploy
gcloud builds submit --tag gcr.io/your-project-id/pothole-detector
gcloud run deploy --image gcr.io/your-project-id/pothole-detector --platform managed --region us-central1 --allow-unauthenticated --memory 2Gi --cpu 2

# Set environment variables
gcloud run services update pothole-detector --set-env-vars="DB_HOST=your-postgres-host,DB_PASSWORD=your-password"
```

### 2. **Railway** (Easy Setup)

**Free Tier**: $5 credit monthly
**Resources**: 512MB RAM, 1GB storage
**Pros**:

- Very easy deployment
- Built-in PostgreSQL
- GitHub integration

**Deployment Steps**:

1. Go to railway.app
2. Connect GitHub repository
3. Add PostgreSQL service
4. Deploy automatically

### 3. **Fly.io** (Performance)

**Free Tier**: 3 small VMs, 256MB RAM each
**Resources**: Can combine for more RAM
**Pros**:

- Good performance
- Global deployment
- Docker support

**Deployment Steps**:

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login and launch
fly auth login
fly launch

# Deploy
fly deploy
```

### 4. **Render** (Current Platform)

**Free Tier**: 750 hours/month
**Resources**: 512MB RAM
**Pros**:

- Simple setup
- Good documentation
  **Cons**:
- Limited RAM for ML models
- Cold starts

## Database Options

### Free PostgreSQL Hosting:

1. **Railway PostgreSQL** - Free tier available
2. **Supabase** - Free tier with 500MB
3. **Neon** - Free tier with 3GB
4. **Google Cloud SQL** - Free tier with 1GB

## Optimization Tips

### For Free Tiers:

1. **Use CPU-only PyTorch** (already configured)
2. **Implement model caching** (already done)
3. **Use background loading** (already implemented)
4. **Optimize image sizes** before processing
5. **Use CDN** for static files

### Memory Optimization:

```python
# Add to app.py for better memory management
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Recommended Deployment Strategy

1. **Start with Railway** - Easiest setup
2. **Move to Google Cloud Run** - Better performance and resources
3. **Use Supabase** - For free PostgreSQL database

## Cost Estimation (Free Tiers)

- **Railway**: $0 (within free credits)
- **Google Cloud Run**: $0 (within free tier)
- **Fly.io**: $0 (within free tier)
- **Database**: $0 (Supabase/Neon free tier)

## Next Steps

1. Choose a platform
2. Set up PostgreSQL database
3. Update environment variables
4. Deploy using provided configurations
5. Test the deployment
