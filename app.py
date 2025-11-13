import streamlit as st
import pandas as pd
import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
import numpy as np
import time


st.set_page_config(
    page_title="RAG Model Evaluation",
    page_icon="",
    layout="wide"
)

st.title("RAG Model Chat Log Evaluation")
st.markdown("Upload your chat log file to evaluate the model using RAGAS metrics")

# Sidebar for configuration
st.sidebar.header("Configuration")

# API Key input
api_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["OpenAI", "Groq", "Custom OpenAI-Compatible"],
    index=0,
    help="OpenAI provides reliable API access. Choose Groq for free tier or Custom for other OpenAI-compatible APIs"
)

if api_provider == "Groq":
    api_key = st.sidebar.text_input(
        "Enter Groq API Key",
        type="password",
        help="Get your free API key from https://console.groq.com"
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    model_name = st.sidebar.selectbox(
        "Select Model",
        [
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=0,
        help="openai/gpt-oss-120b is a powerful 120B parameter open-source model"
    )
    use_custom_endpoint = False
    custom_base_url = None
elif api_provider == "OpenAI":
    api_key = st.sidebar.text_input(
        "Enter OpenAI API Key",
        type="password",
        help="Get your API key from https://platform.openai.com"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
        index=1
    )
    use_custom_endpoint = False
    custom_base_url = None
elif api_provider == "Custom OpenAI-Compatible":
    api_key = st.sidebar.text_input(
        "Enter API Key",
        type="password",
        help="Enter your API key for the custom endpoint"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    use_custom_endpoint = True
    custom_base_url = st.sidebar.text_input(
        "API Base URL",
        value="https://api.together.xyz/v1",
        help="Enter the base URL for your OpenAI-compatible API (e.g., Together AI, OSS models)"
    )
    custom_model = st.sidebar.text_input(
        "Model Name",
        value="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Enter the exact model name"
    )
    if custom_model:
        model_name = custom_model
    else:
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
else:
    use_custom_endpoint = False
    custom_base_url = None
    model_name = "gpt-3.5-turbo"

st.sidebar.markdown("""
### Metrics Evaluated:
- **Faithfulness**: How factually accurate the answers are
- **Answer Relevancy**: How relevant the answers are to the questions
- **Answer Correctness**: Overall correctness of the answers
""")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Chat Log File (JSON or CSV)",
    type=['json', 'csv'],
    help="Upload a file containing questions, answers, contexts, and ground truth"
)

if uploaded_file is not None:
    try:
        # Read the uploaded file
        file_type = uploaded_file.name.split('.')[-1]
        
        if file_type == 'json':
            data = json.load(uploaded_file)
            df = pd.DataFrame(data)
        elif file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        
        st.success(f"File uploaded successfully! Found {len(df)} entries.")
        
        # Display uploaded data
        with st.expander("View Uploaded Data"):
            st.dataframe(df)
        
        # Validate required columns
        required_columns = ['question', 'answer', 'contexts', 'ground_truth']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.info("""
            **Required format:**
            - `question`: The user's question
            - `answer`: The model's answer
            - `contexts`: List of retrieved contexts (can be a list or JSON string)
            - `ground_truth`: The correct/expected answer
            """)
        else:
            # Process contexts column if it's a string
            def process_contexts(x):
                if isinstance(x, str):
                    # Try to parse JSON string
                    try:
                        parsed = json.loads(x)
                        if isinstance(parsed, list):
                            return [str(item) for item in parsed]
                        else:
                            return [str(parsed)]
                    except:
                        # If not JSON, treat as a single context
                        return [str(x)]
                elif isinstance(x, list):
                    # Ensure all items in list are strings
                    return [str(item) for item in x]
                else:
                    # Convert to string and wrap in list
                    return [str(x)]
            
            df['contexts'] = df['contexts'].apply(process_contexts)
            
            # Ensure all other columns are strings
            df['question'] = df['question'].astype(str)
            df['answer'] = df['answer'].astype(str)
            df['ground_truth'] = df['ground_truth'].astype(str)
            
            # Evaluate button
            if st.button("Run RAGAS Evaluation", type="primary"):
                if not api_key:
                    st.error(f"Please enter your {api_provider} API key in the sidebar")
                else:
                    with st.spinner("Evaluating with RAGAS... This may take a few minutes."):
                        try:
                            eval_start = time.time()
                            # Validate contexts are lists of strings
                            for idx, contexts in enumerate(df['contexts']):
                                if not isinstance(contexts, list):
                                    st.error(f"Row {idx}: contexts must be a list, got {type(contexts)}")
                                    raise ValueError(f"Invalid contexts format at row {idx}")
                                for ctx in contexts:
                                    if not isinstance(ctx, str):
                                        st.error(f"Row {idx}: each context must be a string, got {type(ctx)}")
                                        raise ValueError(f"Invalid context item at row {idx}")
                            
                            # Create dataset dict with proper format
                            data_dict = {
                                'question': df['question'].tolist(),
                                'answer': df['answer'].tolist(),
                                'contexts': df['contexts'].tolist(),
                                'ground_truth': df['ground_truth'].tolist()
                            }
                            
                            # Convert to RAGAS dataset format
                            dataset = Dataset.from_dict(data_dict)
                            
                            # Initialize embeddings using RAGAS with LangChain
                            from langchain_community.embeddings import HuggingFaceEmbeddings
                            ragas_embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
                            )
                            
                            # Configure LLM based on provider
                            if api_provider == "Groq":
                                # Use LangChain ChatGroq for RAGAS
                                from langchain_groq import ChatGroq
                                evaluator_llm = ChatGroq(
                                    model=model_name,
                                    api_key=api_key,
                                    temperature=0
                                )
                                
                                # Run evaluation with Groq
                                result = evaluate(
                                    dataset,
                                    metrics=[
                                        faithfulness,
                                        answer_relevancy,
                                        answer_correctness
                                    ],
                                    llm=evaluator_llm,
                                    embeddings=ragas_embeddings,
                                )
                            elif api_provider == "Custom OpenAI-Compatible":
                                # Use custom OpenAI-compatible endpoint
                                from langchain_openai import ChatOpenAI
                                evaluator_llm = ChatOpenAI(
                                    model=model_name,
                                    api_key=api_key,
                                    base_url=custom_base_url,
                                    temperature=0
                                )
                                
                                # Run evaluation with custom endpoint
                                result = evaluate(
                                    dataset,
                                    metrics=[
                                        faithfulness,
                                        answer_relevancy,
                                        answer_correctness
                                    ],
                                    llm=evaluator_llm,
                                    embeddings=ragas_embeddings
                                )
                            else:
                                # Standard OpenAI
                                from langchain_openai import ChatOpenAI
                                evaluator_llm = ChatOpenAI(
                                    model=model_name,
                                    api_key=api_key,
                                    temperature=0
                                )
                                
                                # Run evaluation with OpenAI
                                result = evaluate(
                                    dataset,
                                    metrics=[
                                        faithfulness,
                                        answer_relevancy,
                                        answer_correctness
                                    ],
                                    llm=evaluator_llm,
                                    embeddings=ragas_embeddings
                                )
                            eval_end = time.time()
                            eval_duration = eval_end - eval_start
                            st.success("Evaluation completed!")
                            st.info(f"Evaluation time: {eval_duration:.2f} seconds")
                            
                            # Display metrics
                            st.header("Evaluation Results")
                            
                            # Create columns for metrics
                            col1, col2, col3 = st.columns(3)
                            
                            # Helper to get mean if value is a list
                            faithfulness_val = result['faithfulness']
                            if isinstance(faithfulness_val, list):
                                faithfulness_val = np.mean(faithfulness_val)
                            answer_relevancy_val = result['answer_relevancy']
                            if isinstance(answer_relevancy_val, list):
                                answer_relevancy_val = np.mean(answer_relevancy_val)
                            answer_correctness_val = result['answer_correctness']
                            if isinstance(answer_correctness_val, list):
                                answer_correctness_val = np.mean(answer_correctness_val)

                            with col1:
                                st.metric(
                                    label="Faithfulness",
                                    value=f"{faithfulness_val:.4f}",
                                    help="Measures factual consistency of the answer with the context"
                                )
                            
                            with col2:
                                st.metric(
                                    label="Answer Relevancy",
                                    value=f"{answer_relevancy_val:.4f}",
                                    help="Measures how relevant the answer is to the question"
                                )
                            
                            with col3:
                                st.metric(
                                    label="Answer Correctness",
                                    value=f"{answer_correctness_val:.4f}",
                                    help="Measures overall correctness compared to ground truth"
                                )
                            
                            # Display detailed results
                            st.subheader("Detailed Results")
                            
                            # Convert results to dataframe
                            results_df = result.to_pandas()
                            
                            # Display all metrics
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="ragas_evaluation_results.csv",
                                mime="text/csv"
                            )
                            
                            # Visualizations
                            st.subheader("Metrics Visualization")
                            
                            # Bar chart of average metrics
                            metrics_data = {
                                'Metric': ['Faithfulness', 'Answer Relevancy', 'Answer Correctness'],
                                'Score': [
                                    result['faithfulness'],
                                    result['answer_relevancy'],
                                    result['answer_correctness']
                                ]
                            }
                            metrics_chart_df = pd.DataFrame(metrics_data)
                            st.bar_chart(metrics_chart_df.set_index('Metric'))
                            
                        except Exception as e:
                            import traceback
                            st.error(f"Error during evaluation: {str(e)}")
                            
                            # Show full traceback in expander for debugging
                            with st.expander("View Full Error Details"):
                                st.code(traceback.format_exc())
                            
                            st.info(f"Make sure you have set up your {api_provider} API key correctly and have sufficient credits.")
                            if api_provider == "Groq":
                                st.code("""
# Get your free Groq API key from:
# https://console.groq.com
                                """)
                            else:
                                st.code("""
# Set your OpenAI API key:
# export OPENAI_API_KEY='your-api-key'
# or in Windows PowerShell:
# $env:OPENAI_API_KEY='your-api-key'
                                """)
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please ensure your file is properly formatted.")

else:
    # Show sample format
    st.info("👈 Please upload a chat log file to begin evaluation")
    
    st.subheader("Expected File Format")
    st.markdown("""
    Your file should contain the following columns:
    - **question**: The user's question
    - **answer**: The RAG model's generated answer
    - **contexts**: Retrieved context passages (as a list)
    - **ground_truth**: The correct/expected answer
    """)
    
    # Sample data
    sample_data = {
        "question": ["What is the capital of France?", "Who wrote Romeo and Juliet?"],
        "answer": ["The capital of France is Paris.", "William Shakespeare wrote Romeo and Juliet."],
        "contexts": [
            '["Paris is the capital and most populous city of France."]',
            '["William Shakespeare was an English playwright and poet."]'
        ],
        "ground_truth": ["Paris", "William Shakespeare"]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df)
    
    # Download sample
    sample_csv = sample_df.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_csv,
        file_name="sample_chat_log.csv",
        mime="text/csv"
    )

# Add OpenAI API test button
st.sidebar.subheader("🔌 Test LLM API Connectivity")
if st.sidebar.button("Test LLM API"):
    if not api_key:
        st.sidebar.error(f"Please enter your {api_provider} API key above.")
    else:
        try:
            if api_provider == "Groq":
                from langchain_groq import ChatGroq
                test_llm = ChatGroq(
                    model=model_name,
                    api_key=api_key,
                    temperature=0
                )
                response = test_llm.invoke("Hello, Groq! Are you working?")
                st.sidebar.success(f"Groq API response: {response}")
            else:
                from langchain_openai import ChatOpenAI
                test_llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=api_key,
                    temperature=0
                )
                response = test_llm.invoke("Hello, OpenAI! Are you working?")
                st.sidebar.success(f"OpenAI API response: {response}")
        except Exception as e:
            st.sidebar.error(f"LLM API test failed: {str(e)}")
