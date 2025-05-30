
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
from crewai import Agent, Task 
import re
# Add this near the top of your file, after imports
import os
port = int(os.environ.get("PORT", 10000))

# Replace the st.set_page_config line with:
st.set_page_config(
    page_title="SmartFitAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

AVAILABLE_MODELS = {
    "Gemini 1.5 Flash": {"provider": "google", "model": "gemini-1.5-flash"},
    "LLaMA 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "Mixtral 8x7B": {"provider": "groq", "model": "mixtral-8x7b-32768"},
    "LLaMA 3.3 70B-Versatile": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "LLaMA 3.3 70B SpecDec": {"provider": "groq", "model": "llama-3.3-70b-specdec"},
    "Mistral-Saba-24B": {"provider": "groq", "model": "mistral-saba-24b"},
    "DeepSeek R1 Distill Qwen 32B": {"provider": "groq", "model": "deepseek-r1-distill-qwen-32b"},
    "DeepSeek R1 Distill LLaMA 70B": {"provider": "groq", "model": "deepseek-r1-distill-llama-70b"}
}

def get_model_response(input_text, pdf_content, prompt, model_info):
    st.write("Debug: PDF Content Received:", pdf_content is not None)
    if not pdf_content:
        return "Error: No resume content provided."

    pdf_data = pdf_content[0]["data"] if pdf_content else ""
    pdf_mime = pdf_content[0]["mime_type"] if pdf_content else ""

    full_input = f"{input_text}\n\nResume Content (Base64 Image): {pdf_data[:100]}...\n\n{prompt}"

    if model_info["provider"] == "google":
        model = genai.GenerativeModel(model_info["model"])
        try:
            response = model.generate_content([
                input_text,
                {"mime_type": pdf_mime, "data": pdf_data},
                prompt
            ])
            return response.text
        except Exception as e:
            return f"Error processing resume with Gemini: {str(e)}"
    elif model_info["provider"] == "groq":
        response = groq_client.chat.completions.create(
            model=model_info["model"],
            messages=[{"role": "user", "content": full_input}],
            max_tokens=1024
        )
        return response.choices[0].message.content

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
            st.write("Debug: PDF Successfully Processed")
            return pdf_parts
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None
    else:
        raise FileNotFoundError("No file uploaded")

# Custom Agent Executor
def execute_task(agent, task, input_text, pdf_content, model_info):
    prompt = task.description
    response = get_model_response(input_text, pdf_content, prompt, model_info)
    return response

# Dummy LLM to satisfy CrewAI
class DummyLLM:
    def bind(self, **kwargs):
        return self
    def __call__(self, *args, **kwargs):
        return "Dummy response"

# CrewAI Agents
def create_agents():
    dummy_llm = DummyLLM()
    
    resume_analyzer = Agent(
        role="Resume Analyzer",
        goal="Analyze the resume and provide a detailed assessment",
        backstory="An experienced HR manager skilled in resume evaluation.",
        verbose=True,
        allow_delegation=False,
        llm=dummy_llm
    )

    interview_preparer = Agent(
        role="Interview Preparer",
        goal="Generate technical and behavioral interview questions",
        backstory="An expert in preparing candidates for job interviews.",
        verbose=True,
        allow_delegation=False,
        llm=dummy_llm
    )

    suggestion_generator = Agent(
        role="Suggestion Generator",
        goal="Provide actionable resume improvement suggestions",
        backstory="A resume optimization expert with ATS knowledge.",
        verbose=True,
        allow_delegation=False,
        llm=dummy_llm
    )

    job_fit_scorer = Agent(
        role="Job Fit Scorer",
        goal="Evaluate job fit and provide a score",
        backstory="An ATS scanner expert in job fit analysis.",
        verbose=True,
        allow_delegation=False,
        llm=dummy_llm
    )

    return resume_analyzer, interview_preparer, suggestion_generator, job_fit_scorer

# Tasks
def create_tasks(input_text, pdf_content, agents):
    resume_analyzer, interview_preparer, suggestion_generator, job_fit_scorer = agents

    resume_task = Task(
        description=f"""
        Analyze the resume against the job description: {input_text}.
        Provide a detailed assessment including strengths, weaknesses, areas of improvement, and other relevant factors.
        Use the provided resume content to inform your analysis.
        """,
        agent=resume_analyzer,
        expected_output="A detailed resume analysis report."
    )

    interview_task = Task(
        description=f"""
        Based on the job description: {input_text} and resume content, generate 5 technical questions related to projects and 5 behavioral questions specific to the role and industry.
        Use the provided resume content to tailor the questions.
        """,
        agent=interview_preparer,
        expected_output="A list of 10 interview questions."
    )

    suggestion_task = Task(
        description=f"""
        Provide actionable suggestions to improve the resume for ATS compatibility and human review based on the job description: {input_text}.
        Use the provided resume content to identify specific areas for enhancement.
        """,
        agent=suggestion_generator,
        expected_output="A list of resume improvement suggestions."
    )

    job_fit_task = Task(
        description=f"""
        Evaluate the resume against the job description: {input_text}.
        Provide a job fit score (0-100), key strengths, and areas for improvement based on the provided resume content.
        Format the output as: "Job Fit Score: [number]\nKey Strengths: ...\nAreas for Improvement: ..."
        """,
        agent=job_fit_scorer,
        expected_output="A job fit score with feedback."
    )

    return resume_task, interview_task, suggestion_task, job_fit_task

# Streamlit UI
st.set_page_config(page_title="SmartFitAI", page_icon="🤖", layout="wide")

# Enhanced CSS with Animations
st.markdown("""
    <style>
        .header { text-align: center; color: #00acc1; animation: fadeIn 1s; }
        .sub-header { text-align: center; color: #666; animation: slideIn 1.5s; }
        .info-card { background: #00acc1; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.3s; }
        .info-card:hover { transform: scale(1.05); }
        .stButton>button { background-color: #00acc1; color: white; border-radius: 5px; transition: all 0.3s; }
        .stButton>button:hover { background-color: #00acc1; transform: translateY(-2px); }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const elements = document.querySelectorAll('.info-card');
            elements.forEach(el => {
                el.addEventListener('mouseenter', () => el.style.transform = 'scale(1.05)');
                el.addEventListener('mouseleave', () => el.style.transform = 'scale(1)');
            });
        });
    </script>
""", unsafe_allow_html=True)

st.markdown("""
    <div>
        <h1 class="header">SmartFitAI 🤖</h1>
        <p class="sub-header">Your Intelligent Job Match & Prep Companion!</p>
        <hr>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("ats.png")
    st.header("Upload & Configure")
    input_text = st.text_area("Paste Job Description Here 👇", key="input", placeholder="Paste the job description...")
    uploaded_file = st.file_uploader("Upload Your Resume PDF", type=["pdf"])
    model_choice = st.selectbox("Choose AI Model", list(AVAILABLE_MODELS.keys()), help="Switch between available models.")

# Main Content
if uploaded_file is not None:
    st.success("PDF Uploaded Successfully ✅")
    pdf_content = input_pdf_setup(uploaded_file)
    if pdf_content is None:
        st.error("Failed to process the resume. Please try again.")
    else:
        model_info = AVAILABLE_MODELS[model_choice]

        # Create Agents and Tasks
        agents = create_agents()
        tasks = create_tasks(input_text, pdf_content, agents)

        # Tab Navigation
        selected_tab = option_menu(
            menu_title=None,
            options=["Resume Analysis", "Interview Prep", "Suggestions", "Job Fit"],
            icons=["file-text", "mic", "lightbulb", "star"],
            orientation="horizontal",
            styles={"nav-link-selected": {"background-color": "#00acc1"}}
        )

        progress_bar = st.progress(0)
        response_placeholder = st.container()

        if selected_tab == "Resume Analysis":
            if st.button("Analyze Resume", key="analyze"):
                with st.spinner("Analyzing Resume..."):
                    progress_bar.progress(25)
                    result = execute_task(agents[0], tasks[0], input_text, pdf_content, model_info)
                    progress_bar.progress(100)
                    st.success("Analysis Complete ✅")
                    with response_placeholder:
                        st.subheader("Resume Analysis:")
                        st.write(result)
                        strengths = result.count("strength")
                        weaknesses = result.count("weakness")
                        fig = px.pie(values=[strengths, weaknesses], names=["Strengths", "Weaknesses"], title="Analysis Breakdown")
                        st.plotly_chart(fig)
                        st.download_button(label="Download Analysis", data=result, file_name="analysis.txt")

        elif selected_tab == "Interview Prep":
            if st.button("Generate Interview Questions", key="interview"):
                with st.spinner("Preparing Interview Questions..."):
                    progress_bar.progress(25)
                    result = execute_task(agents[1], tasks[1], input_text, pdf_content, model_info)
                    progress_bar.progress(100)
                    st.success("Questions Generated ✅")
                    with response_placeholder:
                        st.subheader("Interview Questions:")
                        st.write(result)
                        st.download_button(label="Download Questions", data=result, file_name="questions.txt")

        elif selected_tab == "Suggestions":
            if st.button("Get Improvement Suggestions", key="suggestions"):
                with st.spinner("Generating Suggestions..."):
                    progress_bar.progress(25)
                    result = execute_task(agents[2], tasks[2], input_text, pdf_content, model_info)
                    progress_bar.progress(100)
                    st.success("Suggestions Generated ✅")
                    with response_placeholder:
                        st.subheader("Resume Improvement Suggestions:")
                        st.write(result)
                        st.download_button(label="Download Suggestions", data=result, file_name="suggestions.txt")

        elif selected_tab == "Job Fit":
            if st.button("Calculate Job Fit Score", key="job_fit"):
                with st.spinner("Calculating Job Fit..."):
                    progress_bar.progress(25)
                    result = execute_task(agents[3], tasks[3], input_text, pdf_content, model_info)
                    progress_bar.progress(100)
                    st.success("Job Fit Calculated ✅")
                    with response_placeholder:
                        st.subheader("Job Fit Score:")
                        st.write("Debug: Raw Result:", result)  
                        st.write(result)
                        score_match = re.search(r"Job Fit Score:\s*(\d+)", result, re.IGNORECASE)
                        score = int(score_match.group(1)) if score_match else 50
                        
                        # Fixed gauge chart creation
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Job Fit Score"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#00acc1"},
                                'steps': [
                                    {'range': [0, 33], 'color': "#FF5733"},
                                    {'range': [33, 66], 'color': "#FFC300"},
                                    {'range': [66, 100], 'color': "#33FF57"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 85
                                }
                            }
                        ))
                        st.plotly_chart(fig)
                        st.download_button(label="Download Job Fit Report", data=result, file_name="job_fit.txt")

# Collapsible Info Section
with st.expander("Learn About ATS & Interview Prep", expanded=False):
    st.markdown("""
        <h2 style="text-align:center; color:#0078D7;">What is ATS & How to Prepare?</h2>
        <hr style="border: 1px solid #0078D7;">
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>What is ATS?</h3>
                <p>ATS filters resumes based on keywords and criteria before human review.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="info-card">
                <h3>Why Optimize?</h3>
                <p>Optimized resumes increase your chances of passing ATS and landing interviews.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="info-card">
                <h3>Interview Prep</h3>
                <p>Prepare for technical and behavioral questions tailored to the job.</p>
            </div>
        """, unsafe_allow_html=True)

# Updated Footer
st.markdown("""
    <footer>
        <p>© 2025 SmartFitAI 🤖 - Matching You Smartly to Your Dream Job!</p>
    </footer>
""", unsafe_allow_html=True)