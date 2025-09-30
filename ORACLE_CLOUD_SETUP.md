# Oracle Cloud Free Tier Deployment Guide

Complete guide to deploy your MNIST Classifier on Oracle Cloud's Always Free tier VM.

---

## 📋 What You'll Get

- **FREE forever** VM with 4 ARM CPU cores + 24GB RAM
- Both frontend (Next.js) + backend (FastAPI) on one server
- Always-on, no sleep
- Public URL to share your project

---

## Part 1: Create Oracle Cloud VM

### Step 1: Sign Up

1. Go to [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. Click **"Start for free"**
3. Complete registration (requires credit card for verification, but won't charge)
4. Choose your **home region** carefully (can't change later)
   - Recommend: US East (Ashburn) or US West (Phoenix)

### Step 2: Create VM Instance

1. **Login** to Oracle Cloud Console
2. Click **"Create a VM instance"** or navigate to:
   - Menu → Compute → Instances → Create Instance

3. **Configure Instance:**
   ```
   Name: mnist-classifier
   Compartment: (root) or your compartment
   ```

4. **Placement:**
   - Availability domain: AD-1 (or any available)

5. **Image and Shape:**
   - Click **"Change Image"**
     - Select: **Ubuntu 22.04 Minimal**
     - Click "Select Image"

   - Click **"Change Shape"**
     - Select: **Ampere (ARM-based)**
     - Select: **VM.Standard.A1.Flex**
     - OCPUs: **4**
     - Memory (GB): **24**
     - Click "Select Shape"

6. **Networking:**
   - Leave default VCN and subnet
   - Check: **"Assign a public IPv4 address"** ✅

7. **Add SSH Keys:**
   - Generate SSH key pair (if you don't have one):
     ```bash
     ssh-keygen -t rsa -b 4096 -f ~/.ssh/oracle_vm
     ```
   - Select **"Paste public keys"**
   - Paste contents of `~/.ssh/oracle_vm.pub`

8. **Boot Volume:**
   - Leave default (50GB is fine)

9. Click **"Create"**

### Step 3: Wait for Provisioning

- Status will change from "Provisioning" → "Running" (1-2 minutes)
- If you get **"Out of capacity"** error:
  - Try different availability domain (AD-2, AD-3)
  - Try different region
  - Try again in a few hours/days

### Step 4: Note Your Public IP

- Copy the **Public IP address** (e.g., `123.45.67.89`)
- You'll need this to SSH into the VM

---

## Part 2: Configure Firewall Rules

### Step 5: Open Ingress Rules

1. On your instance page, click on the **Subnet** link
2. Click on the **Default Security List**
3. Click **"Add Ingress Rules"**

**Add these 3 rules:**

**Rule 1 - SSH (already exists):**
```
Source CIDR: 0.0.0.0/0
IP Protocol: TCP
Destination Port Range: 22
Description: SSH
```

**Rule 2 - HTTP:**
```
Source CIDR: 0.0.0.0/0
IP Protocol: TCP
Destination Port Range: 80
Description: HTTP
```
Click "Add Ingress Rules"

**Rule 3 - HTTPS:**
```
Source CIDR: 0.0.0.0/0
IP Protocol: TCP
Destination Port Range: 443
Description: HTTPS
```
Click "Add Ingress Rules"

**Rule 4 - Frontend (Next.js):**
```
Source CIDR: 0.0.0.0/0
IP Protocol: TCP
Destination Port Range: 3000
Description: Next.js Frontend
```
Click "Add Ingress Rules"

**Rule 5 - Backend (FastAPI):**
```
Source CIDR: 0.0.0.0/0
IP Protocol: TCP
Destination Port Range: 8000
Description: FastAPI Backend
```
Click "Add Ingress Rules"

---

## Part 3: Setup VM

### Step 6: SSH into VM

```bash
ssh -i ~/.ssh/oracle_vm ubuntu@YOUR_PUBLIC_IP
```

Replace `YOUR_PUBLIC_IP` with your instance's public IP.

### Step 7: Run Setup Script

Once connected, run the automated setup script:

```bash
# Download and run the setup script
curl -o setup.sh https://raw.githubusercontent.com/YOUR_USERNAME/mnist-linear-classifier/main/setup-vm.sh
chmod +x setup.sh
./setup.sh
```

Or manually run the commands below:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y \
    git \
    curl \
    wget \
    docker.io \
    docker-compose \
    nginx \
    ufw

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu

# Configure firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 3000/tcp  # Next.js
sudo ufw allow 8000/tcp  # FastAPI
sudo ufw --force enable

# Log out and back in for docker group to take effect
exit
```

### Step 8: SSH Back In

```bash
ssh -i ~/.ssh/oracle_vm ubuntu@YOUR_PUBLIC_IP
```

---

## Part 4: Deploy Your Application

### Step 9: Clone Your Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/mnist-linear-classifier.git
cd mnist-linear-classifier
```

### Step 10: Update Configuration

Edit the frontend environment variable:

```bash
# Edit docker-compose.yml
nano docker-compose.yml
```

Find the frontend service and update:
```yaml
environment:
  - NEXT_PUBLIC_API_URL=http://YOUR_PUBLIC_IP:8000
```

Replace `YOUR_PUBLIC_IP` with your actual public IP.

Save and exit (Ctrl+X, Y, Enter)

### Step 11: Build and Run

```bash
# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Wait for services to start (30-60 seconds)
```

### Step 12: Test Your Application

Open in browser:
- **Frontend:** `http://YOUR_PUBLIC_IP:3000`
- **Backend API:** `http://YOUR_PUBLIC_IP:8000`
- **API Docs:** `http://YOUR_PUBLIC_IP:8000/docs`

---

## Part 5: Optional - Setup Nginx Reverse Proxy

### Step 13: Configure Nginx

This allows you to access your app on port 80 (default HTTP):

```bash
sudo nano /etc/nginx/sites-available/mnist
```

Paste this configuration:

```nginx
server {
    listen 80;
    server_name YOUR_PUBLIC_IP;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        rewrite ^/api/(.*)$ /$1 break;
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/mnist /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

Now access your app at: `http://YOUR_PUBLIC_IP` (port 80)

---

## Part 6: Optional - Setup Custom Domain + SSL

### Step 14: Point Domain to VM

1. Buy a domain (Namecheap, Cloudflare, etc.)
2. Add **A record** pointing to your VM's public IP
3. Wait for DNS propagation (5-30 minutes)

### Step 15: Install SSL Certificate (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Update Nginx config
sudo nano /etc/nginx/sites-available/mnist
```

Change `server_name YOUR_PUBLIC_IP;` to `server_name yourdomain.com www.yourdomain.com;`

```bash
# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

Now access via HTTPS: `https://yourdomain.com`

---

## 🔧 Maintenance Commands

### View Logs
```bash
cd ~/mnist-linear-classifier
docker-compose logs -f
docker-compose logs api
docker-compose logs frontend
```

### Restart Services
```bash
docker-compose restart
docker-compose restart api
docker-compose restart frontend
```

### Update Application
```bash
cd ~/mnist-linear-classifier
git pull
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### Check Service Status
```bash
docker-compose ps
```

### Monitor Resources
```bash
htop                    # Install: sudo apt install htop
docker stats
```

---

## 🐛 Troubleshooting

### Can't SSH into VM
- Check security list has port 22 open
- Verify you're using correct SSH key
- Try: `ssh -v -i ~/.ssh/oracle_vm ubuntu@YOUR_PUBLIC_IP`

### Out of Capacity Error
- Try different availability domain
- Try different region during signup
- Keep trying at different times (morning works better)

### Services Won't Start
```bash
# Check logs
docker-compose logs

# Check disk space
df -h

# Check memory
free -h

# Rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Can't Access Frontend
- Check firewall: `sudo ufw status`
- Check service: `docker-compose ps`
- Check logs: `docker-compose logs frontend`
- Verify port 3000 is open in Oracle security list

### API Not Working
- Check if models exist: `ls -la ~/mnist-linear-classifier/outputs/`
- Models might not be in repo - you may need to train first:
  ```bash
  docker-compose run --rm train
  ```

### VM Suspended/Terminated
Oracle may reclaim idle VMs. To prevent:
- Keep services running
- Generate some traffic periodically
- Check emails from Oracle

---

## 💰 Cost Estimate

**Total: $0/month** ✅

Everything runs on Always Free tier:
- VM: Free (4 OCPU, 24GB RAM)
- Storage: Free (up to 200GB)
- Bandwidth: Free (unlimited)
- Public IP: Free

---

## 🎉 You're Done!

Your MNIST Classifier is now:
- ✅ Running 24/7 on a free VM
- ✅ Accessible worldwide
- ✅ No sleep issues
- ✅ Full control

**Share your project:**
- Public URL: `http://YOUR_PUBLIC_IP:3000`
- Or with domain: `https://yourdomain.com`

---

## 📚 Next Steps

1. Add your public URL to GitHub README
2. (Optional) Setup custom domain + SSL
3. (Optional) Add monitoring (Uptime Robot)
4. (Optional) Setup automated backups
5. (Optional) Configure CI/CD for auto-deployment

For detailed Vercel/Render deployment, see [DEPLOYMENT.md](DEPLOYMENT.md)
