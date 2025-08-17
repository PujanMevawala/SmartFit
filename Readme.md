# SmartFitAI 🤖

Your Intelligent Job Match & Prep Companion!

## Features

- **Resume Analysis**: Get detailed assessment of your resume against job descriptions
- **Interview Preparation**: Generate tailored technical and behavioral questions
- **ATS Optimization**: Receive suggestions to improve resume for ATS compatibility
- **Job Fit Scoring**: Get a comprehensive job fit score with detailed feedback
- **Multi-Model AI**: Choose from various AI models (Gemini, LLaMA, Mixtral, etc.)

## Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd SmartFit
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Deployment to Render

### Prerequisites
- GitHub account with your code repository
- Render account
- API keys for Groq and Google Generative AI

### Deployment Steps

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Initial deployment setup"
   git push origin main
   ```

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Sign up/Login with your GitHub account
   - Click "New +" and select "Web Service"

3. **Configure the service**
   - **Name**: `smartfit` (or your preferred name)
   - **Repository**: Select your GitHub repository
   - **Branch**: `main`
   - **Root Directory**: Leave empty (if app is in root)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

4. **Set Environment Variables**
   In Render dashboard, go to your service → Environment:
   - `GROQ_API_KEY`: Your Groq API key
   - `GOOGLE_API_KEY`: Your Google Generative AI API key
   - `PORT`: `8501`

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Your app will be available at the provided URL

## API Keys Setup

### Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up/Login
3. Navigate to API Keys section
4. Create a new API key
5. Copy and use in your environment variables

### Google Generative AI API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Navigate to API Keys
4. Create a new API key
5. Copy and use in your environment variables

## File Structure

```
SmartFit/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment configuration
├── config.toml           # Streamlit configuration
├── smartfit_logo.jpg     # Application logo
├── .env                  # Environment variables (not in git)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Security Notes

- Never commit `.env` files to version control
- API keys are stored as environment variables in Render
- The `.gitignore` file excludes sensitive files from Git

## Troubleshooting

### Common Issues

1. **App not loading**: Check if all environment variables are set in Render
2. **PDF processing errors**: Ensure `poppler-utils` is properly installed
3. **API errors**: Verify your API keys are valid and have sufficient credits

### Local Development Issues

1. **Port conflicts**: Change port in `config.toml` if 8501 is in use
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Environment variables**: Ensure `.env` file exists with proper API keys

## Support

For issues and questions:
- Check the troubleshooting section above
- Review Render deployment logs
- Ensure all dependencies are properly installed

---

© 2025 SmartFitAI 🤖 - Matching You Smartly to Your Dream Job!
