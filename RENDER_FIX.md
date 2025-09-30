# Render Deployment Fix

## Problem
Your Render deployment was timing out during health checks because:
1. PyTorch models take time to load on startup
2. Default health check timeout (30s) is too short
3. Using untrained models (no trained models in repo)

## Solution Applied

### 1. Updated `render.yaml`
- Changed runtime from `python` to `docker` (better control)
- Increased health check thresholds:
  - `healthCheckInterval: 30s`
  - `healthCheckTimeout: 10s`
  - `healthCheckFailureThreshold: 5`
- This gives Render ~2.5 minutes before timing out

### 2. Updated `Dockerfile.api`
- Added `curl` for health checks
- Added 60s `start-period` in HEALTHCHECK
- Use `$PORT` environment variable (Render requirement)
- Upgrade pip to avoid warnings
- Create outputs directories

## Deploy to Render

### Option 1: Using render.yaml (Recommended)

1. **Push changes to GitHub:**
   ```bash
   git add render.yaml Dockerfile.api
   git commit -m "Fix Render deployment health check timeout"
   git push
   ```

2. **Deploy on Render:**
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml`
   - Click "Apply"

### Option 2: Manual Setup

1. **Go to Render Dashboard**
2. **Click "New +" → "Web Service"**
3. **Connect GitHub repository**
4. **Configure:**
   ```
   Name: mnist-classifier-api
   Region: Oregon (US West)
   Branch: main
   Runtime: Docker

   Build Settings:
   - Dockerfile Path: ./Dockerfile.api

   Advanced Settings:
   - Health Check Path: /
   - Health Check Interval: 30s
   - Health Check Timeout: 10s
   - Health Check Failure Threshold: 5

   Plan: Free
   ```
5. **Click "Create Web Service"**

## Expected Deployment Timeline

```
Building...           ⏱️  10-15 minutes (first time)
├─ Install dependencies
├─ Install PyTorch (~700MB)
└─ Build Docker image

Deploying...          ⏱️  1-2 minutes
├─ Upload image
├─ Start container
├─ Load models (untrained)
└─ Health check passes ✅

Total: ~12-17 minutes
```

## Verify Deployment

Once deployed, you'll get a URL like: `https://mnist-classifier-api-XXXX.onrender.com`

**Test endpoints:**
```bash
# Health check
curl https://your-app.onrender.com/

# API docs
open https://your-app.onrender.com/docs

# Models info
curl https://your-app.onrender.com/models
```

## Important Notes

### 1. Models are Untrained
Your deployed app will use **untrained models** because trained model files (`outputs/`) are not in your git repo (they're in `.gitignore`).

**To use trained models, you have 2 options:**

**Option A: Upload pre-trained models**
```bash
# On Render, add environment variable or volume
# Not recommended for free tier (ephemeral storage)
```

**Option B: Train on deployment (one-time)**
```bash
# SSH into Render instance (requires paid plan)
# Or use Render Disks (paid feature)
```

**Option C: Use GitHub LFS or external storage**
```bash
# Store models in S3/Cloudflare R2
# Update api.py to download models on startup
```

For demo purposes, **untrained models work fine** - they'll still make predictions, just with lower accuracy.

### 2. Free Tier Limitations

**Render Free Tier:**
- ✅ 750 hours/month
- ⚠️ Sleeps after 15 min inactivity
- ⚠️ First request after sleep: ~30-60s wake up
- ⚠️ 512MB RAM (might be tight for PyTorch)
- ⚠️ Ephemeral storage (files don't persist)

**If app crashes due to memory:**
- Reduce PyTorch dependencies
- Use CPU-only PyTorch: `torch==2.0.0+cpu`
- Upgrade to paid tier ($7/month for 2GB RAM)

### 3. Cold Starts

Free tier apps sleep after 15 min. To keep awake:
- Use [UptimeRobot](https://uptimerobot.com/) (free) to ping every 5 min
- Upgrade to paid tier ($7/month)

## Troubleshooting

### Still Timing Out?
```yaml
# In render.yaml, increase thresholds even more:
healthCheckInterval: 60s
healthCheckTimeout: 30s
healthCheckFailureThreshold: 10
```

### Out of Memory?
```bash
# Check logs in Render dashboard
# If OOM (Out of Memory), upgrade to paid tier or optimize
```

### Build Taking Too Long?
```bash
# PyTorch is ~700MB, first build takes 10-15 min
# Subsequent builds use cache and are faster (~2-3 min)
```

### Can't Connect Frontend to Backend?
```bash
# Update frontend/.env in Vercel:
NEXT_PUBLIC_API_URL=https://your-app.onrender.com

# Update CORS in api.py:
allow_origins=["https://your-frontend.vercel.app"]
```

## Next Steps

After successful deployment:

1. ✅ Copy your Render URL
2. ✅ Deploy frontend to Vercel with backend URL
3. ✅ Update CORS in `api.py` with Vercel domain
4. ✅ Test end-to-end
5. ⚠️ Consider Oracle Cloud VM if you want always-on free hosting

---

## Alternative: Skip Render, Use Oracle Cloud VM

If Render's free tier limitations are too restrictive:
- **Oracle Cloud**: Always-on, no sleep, 24GB RAM, FREE forever
- **See**: [ORACLE_CLOUD_SETUP.md](ORACLE_CLOUD_SETUP.md)
