#!/bin/bash

# MNIST Classifier - Startup Script
# This script starts both the FastAPI backend and Next.js frontend

echo "🚀 Starting MNIST Classifier..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo -e "${GREEN}✓ Virtual environment found${NC}"
    source venv/bin/activate
fi

# Check if node_modules exists in frontend
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Node modules not found. Installing...${NC}"
    cd frontend && npm install && cd ..
else
    echo -e "${GREEN}✓ Node modules found${NC}"
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    kill $API_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on CTRL+C
trap cleanup INT

# Start FastAPI backend
echo -e "${BLUE}Starting FastAPI backend on http://localhost:8000${NC}"
python api.py &
API_PID=$!

# Wait for API to start
sleep 2

# Check if API is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo -e "${GREEN}✓ API is running${NC}"
else
    echo -e "${YELLOW}⚠ API failed to start${NC}"
fi

# Start Next.js frontend
echo -e "${BLUE}Starting Next.js frontend on http://localhost:3000${NC}"
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 3

echo ""
echo -e "${GREEN}✨ MNIST Classifier is ready!${NC}"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔌 API Docs: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop all services"
echo ""

# Wait for processes
wait $API_PID $FRONTEND_PID