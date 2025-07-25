# SmartFitAI 🤖

An Intelligent Job Match & Prep Companion!

## Overview

SmartFitAI is an intelligent resume analysis tool that helps job seekers optimize their resumes for Applicant Tracking Systems (ATS) and prepare for interviews. The application leverages multiple AI models to provide comprehensive analysis, interview preparation, improvement suggestions, and job fit scoring.

## Live Demo

Check out the live application here: [https://smartfit-sqld.onrender.com/](https://smartfit-sqld.onrender.com/)

## Features

- **Resume Analysis**: Detailed assessment of your resume's strengths and weaknesses
- **Interview Prep**: Generation of tailored technical and behavioral interview questions
- **Improvement Suggestions**: Actionable tips to enhance your resume's ATS compatibility
- **Job Fit Scoring**: Quantitative evaluation of how well your resume matches a job description
- **Multiple AI Models**: Support for various AI models including Gemini, LLaMA, Mixtral, and more

## Requirements

- Python 3.8+
- Streamlit
- PDF2Image and Poppler (for PDF processing)
- Groq API key
- Google Generative AI API key

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/PujanMevawala/SmartFit.git
   cd smartfitai
   ```

2. Create a virtual environment and activate it:

   ```
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:

   ```
   pip install -r req.txt
   ```

4. Install Poppler (required for PDF2Image):

   - **Windows**: Download from [http://blog.alivate.com.au/poppler-windows/](http://blog.alivate.com.au/poppler-windows/) and add to PATH
   - **macOS**: `brew install poppler`
   - **Linux**: `sudo apt-get install poppler-utils`

## Configuration

1. Create a `.env` file in the project root with your API keys:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. Make sure you have an `ats.png` image in the project root for the sidebar logo.

## Usage

1. Run the Streamlit application:

   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Using the application:
   - Paste the job description in the sidebar text area
   - Upload your resume as a PDF file
   - Select your preferred AI model
   - Navigate through the tabs to access different features
   - Click the action buttons to analyze your resume, generate interview questions, get suggestions, or calculate job fit

## Supported AI Models

- Gemini 1.5 Flash (Google)
- LLaMA 3.1 8B (Groq)
- Mixtral 8x7B (Groq)
- LLaMA 3.3 70B-Versatile (Groq)
- LLaMA 3.3 70B SpecDec (Groq)
- Mistral-Saba-24B (Groq)
- DeepSeek R1 Distill Qwen 32B (Groq)
- DeepSeek R1 Distill LLaMA 70B (Groq)

## Project Structure

```
smartfitai/
├── app.py           # Main Streamlit application file
├── ats.png          # Logo image for sidebar
├── .env             # Environment variables (API keys)
├── req.txt          # Required packages
└── README.md        # Project documentation
```

## Dependencies

The project uses the following main libraries:

- `streamlit`: For the web interface
- `python-dotenv`: For environment variable management
- `pdf2image`: For converting PDF resumes to images
- `groq`: Groq API client for LLM access
- `google-generativeai`: Google Generative AI API client
- `streamlit-option-menu`: For enhanced navigation
- `crewai`: For agent-based task execution
- `plotly`: For data visualization
- `langchain`: For LLM orchestration
- `embedchain`: For embedding and retrieval

## Troubleshooting

If you encounter issues with PDF processing:

- Ensure Poppler is correctly installed and accessible in your PATH
- Check that your PDF is not password-protected or corrupted

If you see API errors:

- Verify your API keys in the `.env` file
- Check your internet connection
- Ensure you have sufficient credits/quota on your API accounts

## Contact

For any questions or feedback, please reach out to [pujanmevawala080304@gmail.com](mailto:pujanmevawala080304@gmail.com)
