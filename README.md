# RAG Q&A System

This project implements a Retrieval-Augmented Generation (RAG) system that combines information retrieval with generative AI to answer queries based on a given dataset.

## Features

- PDF Document Processing: Extracts text from multiple PDF files.
- Text Splitting: Divides extracted text into manageable passages.
- Embeddings Generation: Creates vector representations of text passages.
- Efficient Retrieval: Uses FAISS for fast similarity search.
- Generative AI Integration: Leverages Google's Generative AI for answer generation.
- Interactive Query Interface: Allows users to input queries and receive AI-generated answers.

## Tech Stack

- Python 3.x
- Libraries:
  - `re`: For text preprocessing and splitting.
  - `faiss`: Facebook AI Similarity Search for efficient similarity search.
  - `PyPDF2`: For reading and extracting text from PDF files.
  - `numpy`: For numerical operations.
  - `google.generativeai`: Google's Generative AI API for text generation.
  - `sentence_transformers`: For generating text embeddings.

## Key Components

1. **RAG Class**: The core component that integrates all functionalities.
2. **Data Loading**: Processes PDF files and extracts text.
3. **Text Processing**: Splits text into passages for efficient retrieval.
4. **Embedding Generation**: Uses SentenceTransformer to create vector representations.
5. **Indexing**: Utilizes FAISS for creating a searchable index of embeddings.
6. **Retrieval**: Finds relevant passages based on query similarity.
7. **Answer Generation**: Uses Google's Generative AI to produce answers based on retrieved context.

## Usage

The system initializes with specified model names and data paths. Users can then input queries, and the system will retrieve relevant information and generate answers.

## Setup

1. Install required libraries: `pip install faiss-cpu PyPDF2 numpy google-generativeai sentence-transformers`
2. Set up Google Generative AI API key.
3. Prepare PDF documents in the specified data path.
4. Run the script and start querying!

Note: Ensure you have the necessary permissions and API keys to use Google's Generative AI service.

![alt text](https://github.com/PranjalSri108/RAG_model/blob/main/Streamlit_App.png?raw=true)

