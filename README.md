# End-to-End Text Summarization NLP Project

An end-to-end machine learning pipeline for automatic text summarization using fine-tuned PEGASUS model. This project demonstrates the complete workflow from data ingestion to model deployment with a FastAPI web interface.

**Project Link:** [dakshrohit/Text-Summarization-NLP-Project](https://github.com/dakshrohit/Text-Summarization-NLP-Project)

## Features

- **Data Pipeline**: Automatic data ingestion, validation, and transformation
- **Model Training**: Fine-tuned PEGASUS model for abstractive text summarization
- **Model Evaluation**: ROUGE metrics calculation
- **FastAPI Web Interface**: REST API for predictions with interactive documentation
- **Docker Support**: Containerization for easy deployment
- **CI/CD**: GitHub Actions workflow for automated testing and AWS ECR deployment

## Workflows

The project follows a modular architecture:

1. Update `config.yaml` - Configure data paths and model locations
2. Update `params.yaml` - Fine-tune model hyperparameters
3. Update `entity` - Define configuration data classes
4. Update `config/configuration.py` - Implement configuration manager
5. Update `conponents` - Implement data processing and model training
6. Update `pipeline` - Create training pipelines for each stage
7. Update `main.py` - Orchestrate the complete pipeline
8. Update `app.py` - Expose predictions via FastAPI endpoints

## Installation

### Prerequisites
- Python 3.8+
- Conda or virtualenv (recommended)
- CUDA-capable GPU (optional, for faster training)

### STEP 01: Create a Conda Environment

```bash
conda create -n summary python=3.8 -y
conda activate summary
```

### STEP 02: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the complete training pipeline:

```bash
python main.py
```

This will execute all stages:
1. Data ingestion (downloads summarization dataset)
2. Data validation (ensures data quality)
3. Data transformation (tokenization and preprocessing)
4. Model training (fine-tunes PEGASUS model)
5. Model evaluation (calculates ROUGE metrics)

### Running the Web Application

Start the FastAPI server (requires trained model):

```bash
python app.py
```

Then open your browser to: `http://localhost:8080/docs`

You'll see the interactive API documentation where you can:
- Test the `/predict` endpoint with sample text
- View the `/train` endpoint for retraining

### API Endpoints

#### GET `/`
Redirects to Swagger UI documentation

#### GET `/train`
Triggers model retraining. This may take several minutes.

**Response:**
```json
"Training successful !!"
```

#### POST `/predict`
Summarizes the provided text.

**Request:**
```json
{
  "text": "Your long text to summarize..."
}
```

**Response:**
```json
{
  "summary_text": "Summary of the provided text..."
}
```



## Project Structure

```
Text-Summarization-NLP-Project/
├── artifacts/               # Generated model and evaluation artifacts
├── config/                 # Configuration files
│   └── config.yaml        # Data and model paths configuration
├── logs/                  # Application logs
├── research/              # Jupyter notebooks for experimentation
├── src/textSummarizer/    # Main package
│   ├── components/        # Core components (data handling, model training, evaluation)
│   ├── config/           # Configuration management
│   ├── entity/           # Data classes for configuration
│   ├── logging/          # Logging setup
│   ├── pipeline/         # Training pipeline stages
│   └── utils/            # Common utilities
├── app.py                 # FastAPI application
├── main.py               # Training orchestration script
├── params.yaml           # Model hyperparameters
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
└── setup.py              # Package setup configuration
```

## Model Details

- **Base Model**: [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail)
- **Fine-tuning Dataset**: SAMSum (Dialog Summarization)
- **Training Approach**: Transfer learning with Hugging Face Transformers
- **Evaluation Metrics**: ROUGE-1, ROUGE-2, ROUGE-L

## Docker Deployment

Build the Docker image:

```bash
docker build -t text-summarizer .
```

Run the container:

```bash
docker run -p 8080:8080 text-summarizer
```

## AWS Deployment with GitHub Actions

The project includes CI/CD pipeline for automated deployment to AWS ECR. For setup:

1. Create an AWS IAM user with ECR and EC2 full access
2. Create an ECR repository
3. Add GitHub secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
4. Push to the `main` branch to trigger the workflow

See `.github/workflows/main.yaml` for workflow configuration.

## Troubleshooting

### Model Files Not Found
Ensure you've run `python main.py` successfully. The model and tokenizer are saved during training.

### GPU Out of Memory
Reduce `per_device_train_batch_size` in `params.yaml` (currently set to 1)

### Port Already in Use
Change the port in `app.py` or kill the existing process:
```bash
lsof -ti:8080 | xargs kill -9
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Author

**Daksh Rohit Sunkara**
- Email: dakshrohitsunkara@gmail.com
- GitHub: [@dakshrohit](https://github.com/dakshrohit)
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app
