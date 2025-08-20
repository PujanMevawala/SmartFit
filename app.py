import base64
from io import BytesIO  
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import pdf2image
from groq import Groq
import google.generativeai as genai
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.card import card
from streamlit_extras.grid import grid
from crewai import Agent, Task 
import re
import os
import time
import pandas as pd
from openai import OpenAI
import logging
import warnings

# Suppress warnings and errors from litellm and other libraries
warnings.filterwarnings("ignore")
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("crewai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

# Suppress litellm and related library verbosity
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DEBUG"] = "false"

st.set_page_config(
    page_title="SmartFitAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize API clients with error handling
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
# Support both PPLX_API_KEY and PERPLEXITY_API_KEY env vars
perplexity_api_key = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    groq_client = None
    st.warning("⚠️ GROQ_API_KEY not found. Some features may not work.")

if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.warning("⚠️ GOOGLE_API_KEY not found. Some features may not work.")

# Perplexity (OpenAI-compatible) client
if perplexity_api_key:
    try:
        pplx_client = OpenAI(base_url="https://api.perplexity.ai", api_key=perplexity_api_key)
    except Exception:
        pplx_client = None
        st.warning("⚠️ Failed to initialize Perplexity client.")
else:
    pplx_client = None
    st.info("ℹ️ PPLX_API_KEY not found. Perplexity models will be unavailable.")

AVAILABLE_MODELS = {
    # Google
    "Gemini 2.5 Pro": {"provider": "google", "model": "gemini-2.5-pro"},
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    "Gemini 1.5 Flash": {"provider": "google", "model": "gemini-1.5-flash"},

    # Groq
    "LLaMA 4 Maverick 17B": {"provider": "groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct"},
    "LLaMA 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "LLaMA 3.3 70B-Versatile": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "DeepSeek R1 Distill LLaMA 70B": {"provider": "groq", "model": "deepseek-r1-distill-llama-70b"},

    # Perplexity (best models)
    "Perplexity Sonar Reasoning Pro": {"provider": "perplexity", "model": "sonar-reasoning-pro"},
    "Perplexity Sonar Large": {"provider": "perplexity", "model": "sonar-large"}
}

def get_model_response(input_text, pdf_content, prompt, model_info):
    if not pdf_content:
        return "Error: No resume content provided."

    pdf_data = pdf_content[0]["data"] if pdf_content else ""
    pdf_mime = pdf_content[0]["mime_type"] if pdf_content else ""

    # Optimized input for faster processing
    full_input = f"{input_text}\n\nResume Content: {pdf_data[:200]}...\n\n{prompt}"

    if model_info["provider"] == "google":
        model_name = model_info["model"]
        model = genai.GenerativeModel(model_name)
        # Optimized retry with shorter delays for faster response
        last_error = None
        for attempt in range(2):  # Reduced retry attempts
            try:
                response = model.generate_content([
                    input_text,
                    {"mime_type": pdf_mime, "data": pdf_data},
                    prompt
                ], generation_config=genai.types.GenerationConfig(
                    max_output_tokens=2048,  # Increased for complete responses
                    temperature=0.3
                ))
                return response.text
            except Exception as e:
                last_error = e
                time.sleep(0.2 * (2 ** attempt))  # Shorter delays
        
        # Fallback for Gemini transient/internal errors
        try:
            fallback_model_name = "gemini-1.5-flash"
            fallback_model = genai.GenerativeModel(fallback_model_name)
            response = fallback_model.generate_content([
                input_text,
                {"mime_type": pdf_mime, "data": pdf_data},
                prompt
            ], generation_config=genai.types.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.3
            ))
            return response.text
        except Exception as e2:
            return f"Error processing resume with Gemini: {str(last_error)} | Fallback failed: {str(e2)}"
    
    elif model_info["provider"] == "groq":
        if groq_client is None:
            return "GROQ client not initialized."
        try:
            response = groq_client.chat.completions.create(
                model=model_info["model"],
                messages=[{"role": "user", "content": full_input}],
                max_tokens=2048,  # Increased for complete responses
                temperature=0.3,
                timeout=15  # Increased timeout for longer responses
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing resume with Groq: {str(e)}"
    
    elif model_info["provider"] == "perplexity":
        if pplx_client is None:
            return "Perplexity client not initialized or API key missing."
        try:
            response = pplx_client.chat.completions.create(
                model=model_info["model"],
                messages=[{"role": "user", "content": full_input}],
                temperature=0.3,
                max_tokens=2048,  # Increased for complete responses
                timeout=15  # Added timeout for longer responses
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing resume with Perplexity: {str(e)}"
    
    return "Error: Unknown model provider."

# PDF Processing Function
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        try:
            images = pdf2image.convert_from_bytes(uploaded_file.read())
            first_page = images[0]
            img_byte_arr = BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            pdf_parts = [{"mime_type": "image/jpeg", "data": base64.b64encode(img_byte_arr).decode()}]
            return pdf_parts
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    else:
        raise FileNotFoundError("No file uploaded")

# Custom Agent Executor with improved response time
def execute_task(agent, task, input_text, pdf_content, model_info):
    prompt = task.description
    response = get_model_response(input_text, pdf_content, prompt, model_info)
    return response

# Benchmark function for comparing models
def benchmark_models(chosen_models, input_text, pdf_content, prompt):
    """Benchmark multiple models and return performance metrics"""
    
    results = []
    
    for model_name in chosen_models:
        if model_name not in AVAILABLE_MODELS:
            continue
            
        model_info = AVAILABLE_MODELS[model_name]
        
        # Measure response time
        start_time = time.time()
        response = get_model_response(input_text, pdf_content, prompt, model_info)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        output_chars = len(response) if response else 0
        
        results.append({
            'model': model_name,
            'provider': model_info['provider'],
            'latency_ms': round(latency_ms, 1),
            'output_chars': output_chars,
            'status': 'Success' if response and not response.startswith('Error') else 'Error'
        })
    
    return pd.DataFrame(results)

# Dummy LLM to satisfy CrewAI and suppress litellm errors
class DummyLLM:
    def __init__(self):
        self.model_name = "dummy-model"
        self.api_key = "dummy-key"
    
    def bind(self, **kwargs):
        return self
    
    def __call__(self, *args, **kwargs):
        return "Dummy response"
    
    def get_supported_openai_params(self):
        return []
    
    def completion(self, *args, **kwargs):
        return {"choices": [{"message": {"content": "Dummy response"}}]}
    
    def predict(self, text, **kwargs):
        return "Dummy response"
    
    def invoke(self, input_data, **kwargs):
        return "Dummy response"

# CrewAI Agents
def create_agents():
    dummy_llm = DummyLLM()
    
    resume_analyzer = Agent(
        role="Resume Analyzer",
        goal="Analyze the resume and provide a detailed assessment",
        backstory="An experienced HR manager skilled in resume evaluation.",
        verbose=False,  # Disable verbose to reduce litellm errors
        allow_delegation=False,
        llm=dummy_llm
    )

    interview_preparer = Agent(
        role="Interview Preparer",
        goal="Generate resume-based technical interview questions with detailed answer guidance",
        backstory="An expert in preparing candidates for job interviews.",
        verbose=False,  # Disable verbose to reduce litellm errors
        allow_delegation=False,
        llm=dummy_llm
    )

    suggestion_generator = Agent(
        role="Suggestion Generator",
        goal="Provide actionable resume improvement suggestions",
        backstory="A resume optimization expert with ATS knowledge.",
        verbose=False,  # Disable verbose to reduce litellm errors
        allow_delegation=False,
        llm=dummy_llm
    )

    job_fit_scorer = Agent(
        role="Job Fit Scorer",
        goal="Evaluate how well the resume matches the job requirements",
        backstory="A recruitment specialist who assesses candidate-job fit.",
        verbose=False,  # Disable verbose to reduce litellm errors
        allow_delegation=False,
        llm=dummy_llm
    )

    return [resume_analyzer, interview_preparer, suggestion_generator, job_fit_scorer]

# CrewAI Tasks
def create_tasks():
    tasks = [
        Task(
            description="""Analyze the provided resume against the job description with professional formatting. Structure your response as follows:

## 📊 RESUME ANALYSIS REPORT

### 🎯 EXECUTIVE SUMMARY
- Brief 2-3 sentence overview of candidate fit

### ✅ STRENGTHS IDENTIFIED
- **Technical Skills**: List specific technical competencies found
- **Experience Highlights**: Key relevant experiences
- **Educational Background**: Relevant qualifications
- **Domain Expertise**: Industry-specific knowledge

### ⚠️ AREAS FOR IMPROVEMENT
- **Skill Gaps**: Missing technical requirements
- **Experience Gaps**: Lacking relevant experience areas
- **Presentation Issues**: Resume formatting/content issues

### 🔍 DETAILED ASSESSMENT

#### Required Qualifications Match
- Education requirements: [Analysis]
- Technical skills alignment: [Analysis]
- Experience requirements: [Analysis]

#### Skills Alignment Analysis
- Core competencies match: [Analysis]
- Technical stack compatibility: [Analysis]
- Soft skills indicators: [Analysis]

### 💡 KEY RECOMMENDATIONS
1. **Immediate Actions**: Priority improvements
2. **Technical Development**: Skill enhancement suggestions
3. **Experience Building**: Relevant project/experience recommendations

### 📈 OVERALL RATING
- **Match Percentage**: X% overall fit
- **Recommendation**: [Hire/Consider/Develop/Pass] with rationale

Use bullet points, clear headers, and professional language throughout.""",
            agent=create_agents()[0],
            expected_output="A professionally formatted resume analysis report with clear sections, ratings, and actionable recommendations."
        ),
        Task(
            description="""Generate 5 resume-based technical interview questions with detailed answer guidance and example responses. Structure your response as follows:

## 🎯 TECHNICAL INTERVIEW PREPARATION GUIDE

### 📚 RESUME-BASED TECHNICAL QUESTIONS

#### Question 1: [Technical Skill from Resume]
**Question**: [Specific technical question based on skills/experience in resume]

**Answer Guidance**: 
- **Key Points to Cover**: [List 3-4 main points to address]
- **Technical Details**: [Specific technical concepts to mention]
- **Tone**: [Professional/Confident/Enthusiastic]
- **Keywords to Use**: [Important technical terms and buzzwords]
- **Example Response Structure**: [Brief outline of how to structure the answer]
- **What Interviewer is Looking For**: [Technical depth, problem-solving approach, etc.]

**Example Response**: [Provide a detailed example response showing exactly how to answer this question with technical depth, proper structure, and professional tone. Include specific technical details, metrics, and demonstrate problem-solving approach.]

#### Question 2: [Another Technical Skill from Resume]
**Question**: [Specific technical question based on skills/experience in resume]

**Answer Guidance**: 
- **Key Points to Cover**: [List 3-4 main points to address]
- **Technical Details**: [Specific technical concepts to mention]
- **Tone**: [Professional/Confident/Enthusiastic]
- **Keywords to Use**: [Important technical terms and buzzwords]
- **Example Response Structure**: [Brief outline of how to structure the answer]
- **What Interviewer is Looking For**: [Technical depth, problem-solving approach, etc.]

**Example Response**: [Provide a detailed example response showing exactly how to answer this question with technical depth, proper structure, and professional tone. Include specific technical details, metrics, and demonstrate problem-solving approach.]

#### Question 3: [Technical Skill from Resume]
**Question**: [Specific technical question based on skills/experience in resume]

**Answer Guidance**: 
- **Key Points to Cover**: [List 3-4 main points to address]
- **Technical Details**: [Specific technical concepts to mention]
- **Tone**: [Professional/Confident/Enthusiastic]
- **Keywords to Use**: [Important technical terms and buzzwords]
- **Example Response Structure**: [Brief outline of how to structure the answer]
- **What Interviewer is Looking For**: [Technical depth, problem-solving approach, etc.]

**Example Response**: [Provide a detailed example response showing exactly how to answer this question with technical depth, proper structure, and professional tone. Include specific technical details, metrics, and demonstrate problem-solving approach.]

#### Question 4: [Technical Skill from Resume]
**Question**: [Specific technical question based on skills/experience in resume]

**Answer Guidance**: 
- **Key Points to Cover**: [List 3-4 main points to address]
- **Technical Details**: [Specific technical concepts to mention]
- **Tone**: [Professional/Confident/Enthusiastic]
- **Keywords to Use**: [Important technical terms and buzzwords]
- **Example Response Structure**: [Brief outline of how to structure the answer]
- **What Interviewer is Looking For**: [Technical depth, problem-solving approach, etc.]

**Example Response**: [Provide a detailed example response showing exactly how to answer this question with technical depth, proper structure, and professional tone. Include specific technical details, metrics, and demonstrate problem-solving approach.]

#### Question 5: [Technical Skill from Resume]
**Question**: [Specific technical question based on skills/experience in resume]

**Answer Guidance**: 
- **Key Points to Cover**: [List 3-4 main points to address]
- **Technical Details**: [Specific technical concepts to mention]
- **Tone**: [Professional/Confident/Enthusiastic]
- **Keywords to Use**: [Important technical terms and buzzwords]
- **Example Response Structure**: [Brief outline of how to structure the answer]
- **What Interviewer is Looking For**: [Technical depth, problem-solving approach, etc.]

**Example Response**: [Provide a detailed example response showing exactly how to answer this question with technical depth, proper structure, and professional tone. Include specific technical details, metrics, and demonstrate problem-solving approach.]

### 💡 INTERVIEW TIPS
- **Preparation Strategy**: Focus on technical skills mentioned in resume
- **Answer Framework**: Use STAR method for technical scenarios
- **Confidence Indicators**: Demonstrate depth of knowledge
- **Follow-up Questions**: Be prepared for technical deep-dives

### 🎯 SUCCESS METRICS
- Technical accuracy and depth
- Clear communication of complex concepts
- Problem-solving methodology
- Alignment with resume claims

Generate exactly 5 questions specifically based on the technical skills, tools, technologies, and experiences mentioned in the candidate's resume. Focus on practical, real-world technical scenarios that would validate the candidate's claimed expertise. Do not include behavioral questions - only technical questions based on resume content. For each question, provide a comprehensive example response that demonstrates the expected level of technical detail and professional communication.""",
            agent=create_agents()[1],
            expected_output="A comprehensive technical interview preparation guide with 5 resume-based questions, detailed answer guidance, and example responses for each question."
        ),
        Task(
            description="""Provide detailed resume improvement suggestions with professional formatting. Structure your response as follows:

## 💡 RESUME OPTIMIZATION GUIDE

### 🎯 PRIORITY IMPROVEMENTS

#### 🔥 CRITICAL FIXES (Do Immediately)
1. **[Issue]**: Specific problem and exact solution
2. **[Issue]**: Clear action item with example
[List 3-5 critical fixes]

#### ⚡ HIGH-IMPACT CHANGES (Do This Week)
1. **[Enhancement]**: Improvement with before/after example
2. **[Enhancement]**: Specific formatting or content change
[List 4-6 high-impact changes]

### 🎨 FORMATTING & STRUCTURE

#### ATS Optimization
- **Keywords to Add**: [List specific keywords for this role]
- **Section Headers**: Recommended standard headers
- **File Format**: PDF vs Word recommendations
- **Length Guidelines**: Optimal page count and content density

#### Visual Improvements
- **Layout Enhancement**: Spacing, margins, font recommendations
- **Section Organization**: Optimal order and structure
- **Readability**: Font, size, and formatting guidelines

### 📝 CONTENT ENHANCEMENTS

#### Experience Section
- **Action Verb Usage**: Replace weak verbs with strong alternatives
- **Quantification**: Add metrics and numbers where missing
- **Achievement Focus**: Transform duties into accomplishments
- **Relevance Ranking**: Prioritize most relevant experiences

#### Skills Section
- **Technical Skills**: Missing skills to add, skills to emphasize
- **Skill Grouping**: How to categorize technical competencies
- **Proficiency Levels**: How to indicate expertise levels

#### Education & Certifications
- **Relevant Coursework**: What to include/exclude
- **Certification Priorities**: Which certifications to pursue
- **Project Highlights**: Academic/personal projects to feature

### 🎯 ROLE-SPECIFIC OPTIMIZATIONS
- **Industry Keywords**: Specific terms for this role/industry
- **Company Research**: How to tailor for target companies
- **Value Proposition**: Key message to emphasize

### ✅ FINAL CHECKLIST
- [ ] Grammar and spelling check
- [ ] Consistent formatting throughout
- [ ] Contact information updated
- [ ] LinkedIn profile alignment
- [ ] Portfolio/GitHub links included
- [ ] References available upon request

Provide specific examples and before/after comparisons where applicable.""",
            agent=create_agents()[2],
            expected_output="A comprehensive resume optimization guide with prioritized improvements, specific examples, and actionable checklists."
        ),
        Task(
            description="""Evaluate job fit with detailed scoring and professional formatting. Structure your response as follows:

## ⭐ JOB FIT ASSESSMENT REPORT

### 📊 OVERALL COMPATIBILITY SCORE
**Job Fit Score: [X]/100**

### 🎯 DETAILED SCORING BREAKDOWN

#### 🔧 Technical Skills Match (40% Weight)
- **Score**: [X]/40 points
- **Analysis**: Detailed breakdown of technical alignment
- **Key Matches**: List specific technical skills that align
- **Gap Analysis**: Missing technical requirements

#### 💼 Experience Relevance (30% Weight)
- **Score**: [X]/30 points
- **Analysis**: How well experience matches role requirements
- **Relevant Projects**: Specific experiences that add value
- **Experience Gaps**: Areas lacking sufficient background

#### 🎓 Education & Qualifications (20% Weight)
- **Score**: [X]/20 points
- **Analysis**: Educational background alignment
- **Degree Relevance**: How education supports role requirements
- **Additional Qualifications**: Certifications, courses, etc.

#### 🎪 Cultural & Role Fit (10% Weight)
- **Score**: [X]/10 points
- **Analysis**: Soft skills and cultural alignment indicators
- **Leadership Potential**: Evidence of growth capability
- **Team Collaboration**: Indicators of teamwork ability

### 📈 COMPETITIVE ANALYSIS
- **Market Position**: How candidate compares to typical applicants
- **Unique Value**: What sets this candidate apart
- **Risk Assessment**: Potential concerns or red flags

### 🚀 HIRING RECOMMENDATION

#### If Score ≥ 80: **STRONG RECOMMEND**
- Excellent fit with minor gaps
- Ready for immediate contribution
- High potential for success

#### If Score 70-79: **RECOMMEND**
- Good fit with some development needed
- Can contribute with proper onboarding
- Solid potential for growth

#### If Score 60-69: **CONSIDER**
- Moderate fit requiring investment
- May need significant development
- Potential if properly supported

#### If Score < 60: **RECONSIDER**
- Limited fit for current role
- Major gaps in requirements
- Better suited for different position

### 💡 IMPROVEMENT PATHWAYS
1. **Immediate Actions**: What candidate can do now
2. **Short-term Development**: 3-6 month improvement plan
3. **Long-term Growth**: Career development suggestions

### 🎯 FINAL VERDICT
**Recommendation**: [Hire/Consider/Develop/Pass]
**Rationale**: [Detailed explanation of recommendation]
**Timeline**: [Suggested next steps and timeline]

**Job Fit Score: [X]**""",
            agent=create_agents()[3],
            expected_output="A comprehensive job fit assessment with detailed scoring, competitive analysis, and clear hiring recommendations."
        )
    ]
    return tasks

# Enhanced CSS with animations and improved styling
st.markdown("""
    <style>
    /* CSS Variables for consistent theming */
    :root {
        --primary-color: #00acc1;
        --primary-light: #4dd0e1;
        --primary-dark: #0097a7;
        --secondary-color: #ff6f00;
        --success-color: #00acc1;
        --warning-color: #ff9800;
        --error-color: #f44336;
        --text-primary: #212121;
        --text-secondary: #757575;
        --background-light: #fafafa;
        --background-dark: #263238;
        --card-shadow: 0 2px 10px rgba(0,0,0,0.1);
        --card-shadow-hover: 0 4px 20px rgba(0,0,0,0.15);
        --border-radius: 12px;
        --transition: all 0.3s ease;
        --gradient-primary: linear-gradient(135deg, #00acc1 0%, #4dd0e1 100%);
        --gradient-success: linear-gradient(135deg, #00acc1 0%, #4dd0e1 100%);
        --gradient-warning: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%);
    }

    /* Global Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header Styles */
    .header-container {
        background: var(--gradient-primary);
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Logo Section */
    .logo-section {
        background: var(--gradient-primary);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
    }

    .logo-text {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .logo-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    /* Section Headers */
    .section-header {
        color: var(--primary-color);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }

    /* Input Labels */
    .input-label {
        color: var(--primary-color);
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    /* Model Info Card */
    .model-info-card {
        background: rgba(0, 172, 193, 0.1);
        border: 1px solid rgba(0, 172, 193, 0.3);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Modern Button Styling */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--border-radius) !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 172, 193, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 172, 193, 0.4) !important;
        background: linear-gradient(135deg, #0097a7 0%, #00acc1 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 10px rgba(0, 172, 193, 0.3) !important;
    }

    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .stButton > button:hover::before {
        left: 100%;
    }

    /* Success/Error Messages */
    .success-message {
        background: var(--gradient-success);
        color: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: var(--card-shadow);
        animation: slideIn 0.5s ease-out;
    }

    .error-message {
        background: var(--gradient-warning);
        color: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: var(--card-shadow);
        animation: slideIn 0.5s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Modern Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }

    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }

    /* Animated Elements */
    .animated-card {
        animation: fadeInUp 0.6s ease-out;
    }

    .pulse-animation {
        animation: pulse 2s infinite;
    }

    .float-animation {
        animation: float 3s ease-in-out infinite;
    }

    /* Progress Bar Enhancement */
    .stProgress > div > div > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 10px !important;
    }

    /* Tab Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 172, 193, 0.1) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(0, 172, 193, 0.3) !important;
        color: var(--primary-color) !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 172, 193, 0.2) !important;
        transform: translateY(-2px) !important;
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(0, 172, 193, 0.3) !important;
    }

    /* Upload Animation */
    .upload-animation {
        background: var(--gradient-success);
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        text-align: center;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
    }

    .upload-animation::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: uploadShimmer 1.5s infinite;
    }

    @keyframes uploadShimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .upload-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }

    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-10px);
        }
        60% {
            transform: translateY(-5px);
        }
    }

    /* Progress Timeline */
    .progress-timeline {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        border-left: 4px solid var(--primary-color);
    }

    .timeline-step {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-radius: 8px;
        transition: var(--transition);
    }

    .timeline-step.active {
        background: rgba(0, 172, 193, 0.1);
        border-left: 3px solid var(--primary-color);
    }

    .timeline-step.completed {
        background: rgba(76, 175, 80, 0.1);
        border-left: 3px solid var(--success-color);
    }

    .timeline-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        width: 40px;
        text-align: center;
    }

    .timeline-text {
        flex: 1;
        color: var(--text-primary);
        font-weight: 500;
    }

    /* Cards */
    .stCard {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(0,0,0,0.1);
        transition: var(--transition);
    }

    .stCard:hover {
        box-shadow: var(--card-shadow-hover);
        transform: translateY(-2px);
    }

    /* Welcome Screen */
    .welcome-container {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: var(--border-radius);
        margin: 2rem 0;
    }

    .welcome-title {
        color: var(--primary-color);
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .welcome-subtitle {
        color: var(--text-secondary);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .feature-item {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        text-align: center;
        transition: var(--transition);
        border: 1px solid rgba(0,0,0,0.1);
    }

    .feature-item:hover {
        box-shadow: var(--card-shadow-hover);
        transform: translateY(-3px);
    }

    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: var(--primary-color);
    }

    .feature-title {
        color: var(--text-primary);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .feature-description {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Navigation Tabs */
    .nav-link {
        color: var(--text-secondary) !important;
        font-weight: 500;
        transition: var(--transition);
    }

    .nav-link:hover {
        color: var(--primary-color) !important;
    }

    .nav-link-selected {
        color: var(--primary-color) !important;
        font-weight: 600;
        border-bottom: 2px solid var(--primary-color);
    }

    /* Footer */
    .footer {
        background: linear-gradient(135deg, var(--background-dark) 0%, #37474f 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-radius: var(--border-radius);
        border-top: 3px solid var(--primary-color);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .welcome-title {
            font-size: 2rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --background-light: #263238;
        }
        
        .stCard {
            background: #37474f;
            color: white;
        }
        
        .feature-item {
            background: #37474f;
            color: white;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">SmartFitAI</h1>
    <p class="header-subtitle">AI-Powered Job Matching</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo Section with image
    st.markdown("""
        <div style="
            text-align: center; 
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(0, 172, 193, 0.15) 0%, rgba(0, 139, 163, 0.15) 100%);
            border-radius: 12px;
            border: 2px solid #00acc1;
            box-shadow: 0 4px 20px rgba(0, 172, 193, 0.3);
        ">
            <img src="data:image/jpeg;base64,{}" style="
                width: 80px; 
                height: 80px; 
                border-radius: 50%; 
                margin-bottom: 1rem;
                border: 3px solid #00acc1;
                box-shadow: 0 4px 15px rgba(0, 172, 193, 0.5);
            ">
            <h3 style="color: #00acc1; margin: 0; font-size: 1.3rem; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">SmartFitAI</h3>
            <p style="color: #00acc1; margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9; font-weight: 500;">
                AI-Powered Job Matching
            </p>
        </div>
    """.format(base64.b64encode(open("smartfit_logo.jpg", "rb").read()).decode()), unsafe_allow_html=True)
    
    # Configuration Section
    st.markdown("""
        <h3 class="section-header">⚙️ Configuration</h3>
    """, unsafe_allow_html=True)
    
    # Job Description Input
    st.markdown('<p class="input-label">📝 Job Description</p>', unsafe_allow_html=True)
    input_text = st.text_area(
        "Paste the job description here...",
        key="input",
        placeholder="Enter the job description to analyze against your resume...",
        height=150,
        help="Paste the complete job description for accurate analysis",
        label_visibility="collapsed"
    )
    
    # Resume Upload
    st.markdown('<p class="input-label">📄 Resume Upload</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="Upload a PDF resume to analyze",
        label_visibility="collapsed"
    )
    
    # Model Selection
    st.markdown('<p class="input-label">🤖 Select AI Model</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Choose your preferred AI model:",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    model_info = AVAILABLE_MODELS[model_choice]
    
    # Model Info Card
    st.markdown(f"""
        <div class="model-info-card">
            <p style="color: #00acc1; margin: 0; font-weight: 600; font-size: 0.9rem;">
                <strong>Selected Model:</strong> {model_choice}
            </p>
            <p style="color: #00acc1; margin: 0.3rem 0 0 0; font-size: 0.85rem; opacity: 0.8;">
                <strong>Provider:</strong> {model_info['provider'].title()}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer in Sidebar with blue accents
    st.markdown("---")
    st.markdown("""
        <div style="
            text-align: center; 
            padding: 1rem;
            background: rgba(0, 172, 193, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(0, 172, 193, 0.3);
        ">
            <p style="color: #00acc1; font-size: 0.8rem; margin: 0; font-weight: 600;">
                SmartFitAI v2.0
            </p>
            <p style="color: #00acc1; font-size: 0.7rem; margin: 0.2rem 0 0 0; opacity: 0.8;">
                Powered by Advanced AI
            </p>
        </div>
    """, unsafe_allow_html=True)

# Main Content Area
if uploaded_file is not None:
    
    pdf_content = input_pdf_setup(uploaded_file)
    if pdf_content is None:
        st.markdown("""
            <div style="
                background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                margin: 1rem 0;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                border-left: 4px solid #b71c1c;
                text-align: center;
            ">
                <h3 style="margin: 0; font-size: 1.2rem;">❌ Processing Error</h3>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                    Failed to process the resume. Please try again with a different file.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Create Agents and Tasks
        agents = create_agents()
        tasks = create_tasks()

        # Enhanced Tab Navigation with better styling
        st.markdown("### 🎯 Analysis Dashboard")
        st.markdown("Select the type of analysis you'd like to perform:")
        
        selected_tab = option_menu(
            menu_title=None,
            options=["Resume Analysis", "Interview Prep", "Suggestions", "Job Fit", "Benchmarks"],
            icons=["📊", "🎯", "💡", "⭐", "⚡"],
            orientation="horizontal",
            styles={
                "nav-link-selected": {"background-color": "#00acc1", "color": "white"},
                "nav-link": {"font-size": "14px", "font-weight": "500", "color": "#666"}
            }
        )

        # Progress and Response Container
        progress_bar = st.progress(0)
        response_placeholder = st.container()

        if selected_tab == "Resume Analysis":
            st.markdown("## 📊 Resume Analysis")
            st.markdown("Get a comprehensive assessment of your resume against the job description.")
            
            if st.button("🚀 Analyze Resume", key="analyze", use_container_width=True):
                with st.spinner("🔍 Analyzing your resume..."):
                    progress_bar.progress(25)
                    result = execute_task(agents[0], tasks[0], input_text, pdf_content, model_info)
                    progress_bar.progress(100)
                
                st.markdown('<div class="success-message">✅ Analysis Complete!</div>', unsafe_allow_html=True)
                
                with response_placeholder:
                    # Quick Stats Section - Moved above Analysis Results
                    st.markdown("### 📈 Quick Stats")
                    strengths = result.lower().count("strength")
                    weaknesses = result.lower().count("weakness")
                    
                    # Metrics in a more compact layout with blue theme
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div style="
                            background: rgba(0, 172, 193, 0.1);
                            border: 1px solid #00acc1;
                            border-radius: 8px;
                            padding: 1rem;
                            margin-bottom: 1rem;
                            text-align: center;
                        ">
                            <h4 style="color: #00acc1; margin: 0 0 0.5rem 0;">Strengths</h4>
                            <div style="font-size: 2rem; font-weight: bold; color: #00acc1;">{}</div>
                        </div>
                        """.format(strengths), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style="
                            background: rgba(255, 152, 0, 0.1);
                            border: 1px solid #ff9800;
                            border-radius: 8px;
                            padding: 1rem;
                            margin-bottom: 1rem;
                            text-align: center;
                        ">
                            <h4 style="color: #ff9800; margin: 0 0 0.5rem 0;">Areas to Improve</h4>
                            <div style="font-size: 2rem; font-weight: bold; color: #ff9800;">{}</div>
                        </div>
                        """.format(weaknesses), unsafe_allow_html=True)
                    
                    with col3:
                        # Chart with better sizing
                        fig = px.pie(
                            values=[strengths, weaknesses],
                            names=["Strengths", "Areas to Improve"],
                            title="Analysis Breakdown",
                            color_discrete_sequence=["#00acc1", "#ff9800"]
                        )
                        fig.update_layout(
                            showlegend=True, 
                            height=200,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Analysis Results - Full width like other tabs
                    st.markdown("### 📋 Analysis Results")
                    # Clean markdown formatting and display as plain text
                    cleaned_result = result.replace('#', '').replace('*', '').replace('##', '').replace('###', '').replace('####', '').replace('**', '').replace('__', '')
                    st.markdown(f"""
                    <div class="stCard animated-card">
                        <div style="white-space: pre-line; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
                            {cleaned_result}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download Button
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=result,
                        file_name="resume_analysis.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        elif selected_tab == "Interview Prep":
            st.markdown("## 🎯 Technical Interview Preparation")
            st.markdown("Generate 5 resume-based technical questions first, then add more questions (resume-based or technical depth).")
            
            # Initialize session state for questions
            if 'interview_questions' not in st.session_state:
                st.session_state.interview_questions = ""
            if 'question_count' not in st.session_state:
                st.session_state.question_count = 0
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🎯 Generate 5 Technical Questions", key="interview", use_container_width=True):
                    with st.spinner("🧠 Preparing technical questions..."):
                        progress_bar.progress(25)
                        result = execute_task(agents[1], tasks[1], input_text, pdf_content, model_info)
                        progress_bar.progress(100)
                
                    st.session_state.interview_questions = result
                    st.session_state.question_count = 5
                    st.markdown('<div class="success-message">✅ 5 Technical Questions Generated!</div>', unsafe_allow_html=True)
            
            with col2:
                if st.button("🔄 Generate More Questions (Resume/Technical)", key="more_questions", use_container_width=True, disabled=st.session_state.question_count == 0):
                    with st.spinner("🧠 Generating additional questions..."):
                        progress_bar.progress(25)
                        # Generate additional questions with a modified prompt
                        additional_prompt = """Generate 3 additional technical interview questions. These can be either:
1. Resume-based questions focusing on other technical skills, experiences, or projects mentioned in the resume
2. In-depth technical questions about programming languages, frameworks, or concepts that would test deep knowledge
3. Advanced technical scenarios or problem-solving questions related to the candidate's background

For each question, provide:
- Question: [The technical question]
- Answer Guidance: 
  - Key Points to Cover: [List 3-4 main points to address]
  - Technical Details: [Specific technical concepts to mention]
  - Tone: [Professional/Confident/Enthusiastic]
  - Keywords to Use: [Important technical terms and buzzwords]
  - Example Response Structure: [Brief outline of how to structure the answer]
  - What Interviewer is Looking For: [Technical depth, problem-solving approach, etc.]
- Example Response: [Provide a detailed example response showing exactly how to answer this question with technical depth, proper structure, and professional tone. Include specific technical details, metrics, and demonstrate problem-solving approach.]

Focus on practical, real-world technical scenarios that would validate the candidate's expertise. Ensure each question has a comprehensive example response."""
                        
                        additional_result = get_model_response(input_text, pdf_content, additional_prompt, model_info)
                        progress_bar.progress(100)
                    
                    # Append new questions to existing ones
                    if st.session_state.interview_questions:
                        st.session_state.interview_questions += "\n\n" + additional_result
                        st.session_state.question_count += 3
                    else:
                        st.session_state.interview_questions = additional_result
                        st.session_state.question_count = 3
                    
                    st.markdown('<div class="success-message">✅ 3 Additional Questions Generated (Resume/Technical)!</div>', unsafe_allow_html=True)
            
            # Display questions if available
            if st.session_state.interview_questions:
                with response_placeholder:
                    st.markdown(f"### 📝 Technical Interview Questions ({st.session_state.question_count} questions)")
                    # Clean markdown formatting and display as plain text
                    cleaned_questions = st.session_state.interview_questions.replace('#', '').replace('*', '').replace('##', '').replace('###', '').replace('####', '').replace('**', '').replace('__', '')
                    st.markdown(f"""
                    <div class="stCard animated-card">
                        <div style="white-space: pre-line; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
                            {cleaned_questions}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Download Questions",
                        data=st.session_state.interview_questions,
                        file_name="technical_interview_questions.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        elif selected_tab == "Suggestions":
            st.markdown("## 💡 Resume Improvement Suggestions")
            st.markdown("Get actionable advice to optimize your resume for ATS and human reviewers.")
            
            if st.button("💡 Get Improvement Suggestions", key="suggestions", use_container_width=True):
                with st.spinner("💭 Generating suggestions..."):
                    progress_bar.progress(25)
                    result = execute_task(agents[2], tasks[2], input_text, pdf_content, model_info)
                    progress_bar.progress(100)
                
                st.markdown('<div class="success-message">✅ Suggestions Generated!</div>', unsafe_allow_html=True)
                
                with response_placeholder:
                    st.markdown("### 🚀 Resume Improvement Suggestions")
                    # Clean markdown formatting and display as plain text
                    cleaned_result = result.replace('#', '').replace('*', '').replace('##', '').replace('###', '').replace('####', '').replace('**', '').replace('__', '')
                    st.markdown(f"""
                    <div class="stCard animated-card">
                        <div style="white-space: pre-line; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
                            {cleaned_result}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Download Suggestions",
                        data=result,
                        file_name="resume_suggestions.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        elif selected_tab == "Job Fit":
            st.markdown("## ⭐ Job Fit Assessment")
            st.markdown("Evaluate how well your resume matches the job requirements.")
            
            if st.button("⭐ Calculate Job Fit Score", key="job_fit", use_container_width=True):
                with st.spinner("📊 Calculating job fit..."):
                    progress_bar.progress(25)
                    result = execute_task(agents[3], tasks[3], input_text, pdf_content, model_info)
                    progress_bar.progress(100)
                
                st.markdown('<div class="success-message">✅ Job Fit Calculated!</div>', unsafe_allow_html=True)
                
                with response_placeholder:
                    st.markdown("### 📊 Job Fit Results")
                    
                    # Extract score
                    score_match = re.search(r"Job Fit Score:\s*(\d+)", result, re.IGNORECASE)
                    score = int(score_match.group(1)) if score_match else 50
                        
                    # Score Display
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### 🎯 Overall Score")
                        st.metric(
                            "Job Fit Score",
                            f"{score}/100",
                            delta=f"{score-50:+d}",
                            delta_color="normal" if score >= 70 else "off" if score >= 50 else "inverse"
                        )
                        
                        # Score Interpretation
                        if score >= 80:
                            st.success("🎉 Excellent Match!")
                        elif score >= 70:
                            st.info("👍 Good Match")
                        elif score >= 50:
                            st.warning("⚠️ Moderate Match")
                        else:
                            st.error("❌ Needs Improvement")
                    
                    with col2:
                        # Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Job Fit Score", 'font': {'size': 20}},
                            delta={'reference': 50, 'increasing': {'color': "green"}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "#00acc1"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 33], 'color': "#ff6b6b"},
                                    {'range': [33, 66], 'color': "#ffd93d"},
                                    {'range': [66, 100], 'color': "#6bcf7f"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 85
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed Results
                    st.markdown("### 📋 Detailed Analysis")
                    # Clean markdown formatting and display as plain text
                    cleaned_result = result.replace('#', '').replace('*', '').replace('##', '').replace('###', '').replace('####', '').replace('**', '').replace('__', '')
                    st.markdown(f"""
                    <div class="stCard animated-card">
                        <div style="white-space: pre-line; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
                            {cleaned_result}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Download Job Fit Report",
                        data=result,
                        file_name="job_fit_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        elif selected_tab == "Benchmarks":
            st.markdown("## ⚡ Model Benchmarks")
            st.markdown("Compare performance across different AI models.")
            
            # Model Selection
            default_models = [
                "Gemini 2.5 Flash",
                "Gemini 2.5 Pro", 
                "LLaMA 4 Maverick 17B",
                "Perplexity Sonar Reasoning Pro"
            ]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                chosen_models = st.multiselect(
                    "Select models to benchmark",
                    list(AVAILABLE_MODELS.keys()),
                    default=[m for m in default_models if m in AVAILABLE_MODELS],
                    help="Choose which models to compare"
                )
            
            with col2:
                st.markdown("### 📊 Benchmark Info")
                st.info(f"Selected: {len(chosen_models)} models")
            
            if st.button("⚡ Run Benchmark", key="run_benchmark", use_container_width=True):
                if not chosen_models:
                    st.warning("Please select at least one model to benchmark.")
                else:
                    with st.spinner("⚡ Running benchmarks..."):
                        progress_bar.progress(25)
                        task_prompt = tasks[0].description
                        bench = benchmark_models(chosen_models, input_text, pdf_content, task_prompt)
                        progress_bar.progress(100)
                    
                    st.markdown('<div class="success-message">✅ Benchmark Complete!</div>', unsafe_allow_html=True)
                    
                    # Results Display
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### 📊 Benchmark Results")
                        st.dataframe(
                            bench,
                            use_container_width=True,
                            column_config={
                                "latency_ms": st.column_config.NumberColumn(
                                    "Latency (ms)",
                                    format="%.1f ms"
                                ),
                                "output_chars": st.column_config.NumberColumn(
                                    "Output Size",
                                    format="%d chars"
                                )
                            }
                        )
                    
                    with col2:
                        st.markdown("### 🚀 Performance Charts")
                        
                        # Latency Chart
                        if bench:
                            fig_latency = px.bar(
                                bench,
                                x="model",
                                y="latency_ms",
                                color="provider",
                                title="Response Latency by Model",
                                color_discrete_sequence=["#00acc1", "#ff6b35", "#4caf50"]
                            )
                            fig_latency.update_layout(height=300)
                            st.plotly_chart(fig_latency, use_container_width=True)
                            
                            # Output Size Chart
                            fig_output = px.bar(
                                bench,
                                x="model", 
                                y="output_chars",
                                color="provider",
                                title="Output Size by Model",
                                color_discrete_sequence=["#00acc1", "#ff6b35", "#4caf50"]
                            )
                            fig_output.update_layout(height=300)
                            st.plotly_chart(fig_output, use_container_width=True)

else:
    # Simple upload prompt when no file is uploaded
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            margin: 2rem 0;
            border: 2px solid rgba(0, 172, 193, 0.1);
        ">
            <h2 style="color: #00acc1; margin-bottom: 1rem;">📄 Upload Your Resume</h2>
            <p style="color: #666; font-size: 1.1rem; line-height: 1.6;">
                Please upload your PDF resume in the sidebar to get started with the analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Info Section
with st.expander("📚 Learn About ATS & Interview Prep", expanded=False):
    st.markdown("""
        <h2 style="text-align:center; color:#00acc1; margin-bottom: 2rem;">What is ATS & How to Prepare?</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>🔍 What is ATS?</h3>
                <p>Applicant Tracking Systems filter resumes based on keywords and criteria before human review.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card">
                <h3>⚡ Why Optimize?</h3>
                <p>Optimized resumes increase your chances of passing ATS and landing interviews.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-card">
                <h3>🎯 Interview Prep</h3>
                <p>Prepare for resume-based technical questions with detailed answer guidance.</p>
            </div>
""", unsafe_allow_html=True)