# MNIST Digit Recognition

This is a beginner-friendly machine learning project that implements a simple neural network using PyTorch to classify handwritten digits from the MNIST dataset.

## Project Features
- **Data Preparation:** Uses `torchvision` to download and preprocess the MNIST dataset.
- **Model Training:** Implements a Convolutional Neural Network (CNN) to train on the dataset.
- **Model Evaluation:** Evaluates the trained model on a test set and prints the accuracy.
- **Containerization:** A Dockerfile is provided to containerize the ML workflow.

## Getting Started

### Prerequisites
- Docker (optional, if you wish to run the project inside a container)
- Python 3.8 or higher (if running locally)
- pip

### Installation

#### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mnist-digit-recognition.git
   cd mnist-digit-recognition

2.	Install the dependencies:
   ```bash
   pip install -r requirements.txt

3.	Run the project:
   ```bash
   python src/main.py

#### Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t mnist-digit-recognition .

2.	Run the Docker container:
   ```bash
   docker run --rm mnist-digit-recognition

3.	Run the project:
   ```bash
   python src/main.py

## Getting Started
```Markdown
mnist-digit-recognition/
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    ├── model.py
    └── train.py

## Getting Started
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- PyTorch
- torchvision
- MNIST Dataset
