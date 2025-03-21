# BERT-based Sentiment Analysis

This project implements a BERT-based sentiment analysis model for analyzing text sentiment. It uses the Hugging Face Transformers library and PyTorch for fine-tuning BERT on sentiment classification tasks.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `custom_bert.py`: Main implementation file containing the BERT model and training pipeline
- `api.py`: FastAPI application for serving the model predictions
- `datasets/`: Directory containing the dataset files
- `.cache/`: Directory for storing processed data and model checkpoints
- `saved_model/`: Directory containing the trained model and tokenizer

## Features

- Fine-tuning BERT for sentiment classification
- Hyperparameter optimization using Optuna
- Support for English language detection
- Comprehensive evaluation metrics and visualizations
- Caching mechanism for processed data
- Early stopping and model checkpointing
- REST API for model deployment

## Usage

### Training the Model

1. Prepare your dataset in CSV format with columns: 'review_title', 'review_text', and 'class_index'
2. Place your dataset in the `datasets/` directory
3. Run the training script:
```bash
python custom_bert.py
```

### Using the API

1. Start the API server:
```bash
python api.py
```

2. The API will be available at `http://localhost:8000`

3. API Endpoints:
   - POST `/predict`: Get sentiment prediction for a review
   - GET `/health`: Check API health status

4. Example API request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"review_title": "Great product!", "review_text": "I really enjoyed using this product. It exceeded my expectations."}'
```

5. Example response:
```json
{
    "sentiment": "positive",
    "confidence": 0.95,
    "probabilities": {
        "positive": 0.95,
        "negative": 0.05
    },
    "is_english": true
}
```

## Model Details

- Base model: bert-base-uncased
- Task: Binary sentiment classification (Positive/Negative)
- Training features: Combined review title and text
- Optimization: Gradient checkpointing for memory efficiency
- Evaluation metrics: Classification report, confusion matrix

## Output

The model training process will generate:
- Training loss plots
- Confusion matrix visualization
- Classification report
- Saved model checkpoints

## Requirements

- Python 3.9+
- PyTorch
- Transformers
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- FastAPI
- Uvicorn
- CUDA (optional, for GPU acceleration)

## Deployment

The API can be deployed to various cloud platforms:

1. **Heroku**:
   - Create a `Procfile` with: `web: uvicorn api:app --host 0.0.0.0 --port $PORT`
   - Deploy using Heroku CLI or GitHub integration

2. **AWS Elastic Beanstalk**:
   - Create a `requirements.txt` file
   - Create a `.ebextensions` directory with configuration files
   - Deploy using AWS CLI or Elastic Beanstalk console

3. **Google Cloud Run**:
   - Create a Dockerfile
   - Build and deploy using Google Cloud CLI

4. **Docker**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt
   EXPOSE 8000
   CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

Remember to:
- Set appropriate environment variables
- Configure CORS if needed
- Set up proper authentication
- Monitor API usage and performance
- Implement rate limiting
- Set up proper logging and monitoring 