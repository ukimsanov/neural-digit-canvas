# Use the official AWS Lambda Python 3.12 base image (2025 latest)
FROM public.ecr.aws/lambda/python:3.12

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source code
COPY api.py ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Copy trained models (only the best models to minimize size)
RUN mkdir -p ${LAMBDA_TASK_ROOT}/outputs/cnn/checkpoints
RUN mkdir -p ${LAMBDA_TASK_ROOT}/outputs/linear/checkpoints

# Copy only the best/final models - not all checkpoints
COPY outputs/cnn/final_model.pth ${LAMBDA_TASK_ROOT}/outputs/cnn/final_model.pth
COPY outputs/cnn/checkpoints/best_model.pth ${LAMBDA_TASK_ROOT}/outputs/cnn/checkpoints/best_model.pth
COPY outputs/linear/final_model.pth ${LAMBDA_TASK_ROOT}/outputs/linear/final_model.pth
COPY outputs/linear/checkpoints/best_model.pth ${LAMBDA_TASK_ROOT}/outputs/linear/checkpoints/best_model.pth

# Set the Lambda function handler
CMD ["api.handler"]