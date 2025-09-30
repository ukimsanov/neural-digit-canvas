#!/bin/bash
# Setup script for Oracle Cloud VM (Ubuntu 22.04 ARM)
# Run this on a fresh Ubuntu instance

set -e  # Exit on error

echo "=================================================="
echo "MNIST Classifier - Oracle Cloud VM Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Update system
echo -e "${GREEN}[1/7] Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Install essential packages
echo -e "${GREEN}[2/7] Installing essential packages...${NC}"
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
echo -e "${GREEN}[3/7] Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Set up the repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker

    echo -e "${GREEN}Docker installed successfully!${NC}"
else
    echo -e "${YELLOW}Docker already installed, skipping...${NC}"
fi

# Add current user to docker group
echo -e "${GREEN}[4/7] Adding user to docker group...${NC}"
sudo usermod -aG docker $USER

# Install Docker Compose (standalone)
echo -e "${GREEN}[5/7] Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose installed successfully!${NC}"
else
    echo -e "${YELLOW}Docker Compose already installed, skipping...${NC}"
fi

# Install Nginx (optional, for reverse proxy)
echo -e "${GREEN}[6/7] Installing Nginx...${NC}"
sudo apt-get install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# Configure firewall
echo -e "${GREEN}[7/7] Configuring firewall (UFW)...${NC}"
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'
sudo ufw allow 3000/tcp comment 'Next.js Frontend'
sudo ufw allow 8000/tcp comment 'FastAPI Backend'
echo "y" | sudo ufw enable

echo ""
echo -e "${GREEN}=================================================="
echo "Setup Complete!"
echo "==================================================${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: You need to log out and log back in for Docker group changes to take effect.${NC}"
echo ""
echo "Next steps:"
echo "1. Log out: exit"
echo "2. SSH back in"
echo "3. Clone your repository:"
echo "   git clone https://github.com/YOUR_USERNAME/mnist-linear-classifier.git"
echo "4. Update docker-compose.yml with your public IP"
echo "5. Run: cd mnist-linear-classifier && docker-compose up -d"
echo ""
echo "Installed versions:"
docker --version
docker-compose --version
nginx -v
echo ""
echo -e "${GREEN}Reboot now? (recommended) [y/N]${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Rebooting in 5 seconds... (Ctrl+C to cancel)"
    sleep 5
    sudo reboot
else
    echo -e "${YELLOW}Please log out and log back in for group changes to take effect.${NC}"
fi
