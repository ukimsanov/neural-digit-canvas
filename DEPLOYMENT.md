# Deployment Guide

This guide explains how to deploy the MNIST Classifier monorepo with Next.js frontend and FastAPI backend.

## Overview

- **Frontend**: Next.js → Deploy to Vercel
- **Backend**: FastAPI → Deploy to Render
- **Architecture**: Frontend calls backend API via environment variable

---

## 1. Deploy Backend to Render

### Steps:

1. **Push to GitHub** (if not already)
   ```bash
   git add .
   git commit -m "Add deployment configs"
   git push origin main
   ```

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up/login with GitHub

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `mnist-classifier-api`
     - **Region**: Oregon (or closest to you)
     - **Branch**: `main`
     - **Root Directory**: Leave blank (or `.`)
     - **Runtime**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
     - **Plan**: Free

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)
   - Copy your backend URL: `https://mnist-classifier-api-XXXX.onrender.com`

### Important Notes:
- Free tier sleeps after 15 min inactivity (first request takes ~30s to wake)
- Models will load on startup
- Health check available at `/` endpoint

---

## 2. Deploy Frontend to Vercel

### Steps:

1. **Create Vercel Account**
   - Go to [vercel.com](https://vercel.com)
   - Sign up/login with GitHub

2. **Import Project**
   - Click "Add New..." → "Project"
   - Import your GitHub repository
   - Configure:
     - **Framework Preset**: Next.js
     - **Root Directory**: `frontend`
     - **Build Command**: `npm run build`
     - **Output Directory**: Leave default (`.next`)
     - **Install Command**: `npm install`

3. **Set Environment Variables**
   - Add environment variable:
     - **Key**: `NEXT_PUBLIC_API_URL`
     - **Value**: `https://mnist-classifier-api-XXXX.onrender.com` (your Render URL)
   - Apply to: Production, Preview, Development

4. **Deploy**
   - Click "Deploy"
   - Wait for deployment (2-3 minutes)
   - Visit your live app: `https://your-project.vercel.app`

---

## 3. Update CORS Settings

After deployment, update your backend CORS settings:

1. **Edit `api.py`** line 58:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=[
           "http://localhost:3000",           # Local development
           "https://your-project.vercel.app",  # Production
           "https://*.vercel.app"              # Preview deployments
       ],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Commit and push** - Render auto-deploys on push

---

## 4. Test Deployment

1. Visit your Vercel URL
2. Check API status indicator in header (should show "Online")
3. Draw a digit and test prediction
4. Monitor Render logs if issues occur

---

## Alternative Deployment Options

### Option 1: Both on Vercel
- Frontend: Vercel (same as above)
- Backend: Vercel Serverless Function (requires code changes, not ideal for ML models)

### Option 2: Railway (Backend)
- Similar to Render, better free tier limits
- Deploy: Connect GitHub → Set start command
- Pricing: $5/month after free tier

### Option 3: Fly.io (Backend)
- Better for global deployments
- Free tier: 3 shared-cpu VMs
- Requires Dockerfile (already exists)

### Option 4: AWS Lambda + API Gateway (Backend)
- Best for pay-per-use
- Requires containerization
- More complex setup

---

## Troubleshooting

### Backend won't start on Render:
- Check build logs for missing dependencies
- Ensure `requirements.txt` includes all packages
- Verify Python version (3.11+)

### Frontend can't connect to backend:
- Verify `NEXT_PUBLIC_API_URL` is set in Vercel
- Check CORS settings in `api.py`
- Inspect Network tab in browser DevTools

### Models not loading:
- Render free tier has limited storage
- You may need to train models after deployment or upload to Render

### Cold starts (Render free tier):
- First request after 15min takes 30-60s
- Consider upgrading to paid tier or use Railway

---

## Cost Estimates

### Free Tier (Recommended for Demo):
- **Vercel**: Free (100GB bandwidth, unlimited deployments)
- **Render**: Free (750 hours/month, sleeps after 15min)
- **Total**: $0/month

### Production Ready:
- **Vercel Pro**: $20/month (team features, analytics)
- **Render Standard**: $7/month (always-on, more resources)
- **Total**: ~$27/month

### High Traffic:
- **Vercel Pro**: $20/month
- **Railway**: $5-20/month (usage-based)
- **Total**: ~$25-40/month

---

## Next Steps

1. Deploy backend to Render
2. Deploy frontend to Vercel
3. Update CORS settings
4. Test the live application
5. (Optional) Set up custom domain
6. (Optional) Add analytics/monitoring

For questions or issues, check the [README.md](README.md) or open an issue on GitHub.
