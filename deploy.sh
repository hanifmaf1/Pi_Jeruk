#!/bin/bash

# Configuration
IMAGE_NAME="orange-detection"
CONTAINER_NAME="orange-app"
PORT=8501

echo "🚀 Starting deployment for $IMAGE_NAME..."

# Build the image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# Check if image build was successful
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed. Exiting."
    exit 1
fi

# Stop and remove existing container if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "🛑 Stopping and removing existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the new container
echo "🏃 Running new container: $CONTAINER_NAME on port $PORT"
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8501 \
    --restart unless-stopped \
    $IMAGE_NAME

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "🌐 You can access the app at: http://localhost:$PORT"
else
    echo "❌ Deployment failed."
    exit 1
fi
