# IMDb Sentiment Analysis API 
[![Continuous Integration](https://github.com/LukeADay/IMDB-Sentiment-Analysis-API/actions/workflows/ci.yml/badge.svg)](https://github.com/LukeADay/IMDB-Sentiment-Analysis-API/actions/workflows/ci.yml) ![CD](https://github.com/LukeADay/IMDB-Sentiment-Analysis-API/actions/workflows/cd.yml/badge.svg)


A project showcasing natural language processing (NLP) using the IMDb movie reviews dataset for binary sentiment classification. This project includes training an LSTM model from scratch, deploying it via a RESTful API, containerizing with Docker, deploying to Kubernetes.

This project uses the IMDb Movie Reviews dataset to classify reviews as positive or negative. Key features include:

* Text Preprocessing: Tokenizes and pads review text.
* Model Training: Builds and trains an LSTM model from scratch.
* REST API: Provides a FastAPI-based endpoint for real-time sentiment prediction.
* Web Application (Flask): Offers a simple UI for users to input movie reviews and receive sentiment predictions.
* Containerisation: Dockerizes the application for easy deployment.
* Kubernetes: Deploys the application to a Kubernetes cluster with autoscaling.


## Dataset
The IMDb dataset consists of 50,000 labeled movie reviews (positive or negative). This dataset is well-suited for binary sentiment analysis tasks.

## Installation
### Prerequisites

* Python 3.8+
* Docker
* Kubernetes
* Setup Instructions

1. Clone the repository:
```
git clone https://github.com/LukeADay/IMDB-Sentiment-Analysis-API.git

cd IMDB-Sentiment-Analysis-API
```

2. Install dependencies:

It’s recommended to set up a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Then install dependencies: `pip install -r requirements.txt`.



3. Download the IMDb dataset:

`python train_model.py --download-dataset`

4. Model Training:

To train the LSTM model, run: `python train_model.py`

This will preprocess the data, train the model, and save `sentiment_model.keras` and `tokenizer.pkl` for deployment.

## API Deployment

### Running the Flask Application:

Run the Flask application to access a simple UI for sentiment prediction:
`python flask_app.py`

The Flask app will be available at http://127.0.0.1:5000.


Or run the FastAPI server locally:

`uvicorn app:app --reload`

The API will be available at http://127.0.0.1:8000.

Example Request:

`curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"text": "I loved the movie!"}'`

For interactive documentation, visit the FastAPI Swagger UI at:

http://127.0.0.1:8000/docs

## Dockerization
Build and run the Docker container:

`
docker build -t imdb-sentiment-api .
docker run -p 8000:8000 imdb-sentiment-api
`

## Kubernetes Deployment

Deploy the application:

`kubectl apply -f deployment.yaml`

Enable autoscaling:

`kubectl autoscale deployment imdb-sentiment-deployment --cpu-percent=50 --min=1 --max=10`

## Project Structure


```
IMDB-Sentiment-Analysis-API/
├── app.py                  # FastAPI application
├── flask_app.py            # Flask application for UI-based sentiment prediction
├── train_model.py          # Training script for LSTM model
├── Dockerfile              # Docker container setup
├── requirements.txt        # Python dependencies
├── deployment.yaml         # Kubernetes deployment 
├── sentiment_model.keras   # Trained model
├── tokenizer.pkl           # Serialized tokenizer
└── templates/
    └── index.html          # HTML template for Flask application
```