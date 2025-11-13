# RAG Model Chat Log Evaluation with RAGAS

A Streamlit application for evaluating Retrieval-Augmented Generation (RAG) model chat logs using RAGAS metrics.

## Features

- 📁 Upload chat log files (JSON or CSV format)
- 📊 Evaluate using RAGAS metrics:
  - **Faithfulness**: Measures factual consistency of answers with context
  - **Answer Relevancy**: Measures how relevant answers are to questions
  - **Answer Correctness**: Measures overall correctness compared to ground truth
- 📈 Visual representation of evaluation results
- 💾 Download evaluation results as CSV

## Installation

1. Install the required dependencies:

```powershell
pip install -r requirements.txt
```

## Configuration

RAGAS requires an LLM for evaluation. You can use **Groq** (recommended for demos - free tier available), **OpenAI**, or any **OpenAI-compatible API** (Together AI, OSS models, etc.).

### Using Groq (Recommended for Demos)

Get a free API key from [Groq Console](https://console.groq.com):

**Available Models:**
- `llama-3.3-70b-versatile` (Latest, recommended)
- `llama-3.1-8b-instant` (Fast)
- `llama3-70b-8192`
- `mixtral-8x7b-32768`
- `gemma2-9b-it`

**Note:** Some older models like `llama-3.1-70b-versatile` have been decommissioned. Use `llama-3.3-70b-versatile` instead.

### Using Groq (Recommended for Demos)

Get a free API key from [Groq Console](https://console.groq.com):

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY='your-groq-api-key-here'
```

**Command Prompt:**
```cmd
set GROQ_API_KEY=your-groq-api-key-here
```

**Linux/Mac:**
```bash
export GROQ_API_KEY='your-groq-api-key-here'
```

### Using OpenAI

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY='your-api-key-here'
```

**Command Prompt:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

### Using Custom OpenAI-Compatible APIs (OSS Models, Together AI, etc.)

You can use any OpenAI-compatible API endpoint. Examples:

**Together AI:**
- Base URL: `https://api.together.xyz/v1`
- Models: `mistralai/Mixtral-8x7B-Instruct-v0.1`, `meta-llama/Llama-3-70b-chat-hf`, etc.

**OpenRouter:**
- Base URL: `https://openrouter.ai/api/v1`
- Models: Various OSS models available

**Local/Self-hosted:**
- Base URL: Your local endpoint (e.g., `http://localhost:8000/v1`)
- Models: Your deployed model name

Simply select "Custom OpenAI-Compatible" in the sidebar and enter your API base URL and model name.

Alternatively, you can enter your API key directly in the Streamlit sidebar when running the application.

## Usage

1. Run the Streamlit application:

```powershell
streamlit run app.py
```

2. Open your browser (usually at `http://localhost:8501`)

3. Select your LLM provider (Groq or OpenAI) from the sidebar

4. Enter your API key in the sidebar

5. Upload your chat log file (JSON or CSV format)

6. Click "Run RAGAS Evaluation" to start the evaluation

7. View and download the results

## Chat Log Format

Your chat log file must contain the following columns:

- **question**: The user's question
- **answer**: The RAG model's generated answer
- **contexts**: Retrieved context passages (as a list)
- **ground_truth**: The correct/expected answer

### Example JSON Format:

```json
[
  {
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "contexts": ["Paris is the capital and most populous city of France."],
    "ground_truth": "Paris"
  }
]
```

### Example CSV Format:

```csv
question,answer,contexts,ground_truth
"What is the capital of France?","The capital of France is Paris.","[""Paris is the capital and most populous city of France.""]","Paris"
```

## Sample File

A sample chat log file (`sample_chat_log.json`) is included in the repository for testing purposes.

## Metrics Explanation

### Faithfulness
Measures the factual consistency of the generated answer against the given context. A score closer to 1 indicates that the answer is more faithful to the context.

### Answer Relevancy
Measures how relevant the generated answer is to the given question. Higher scores indicate better relevance.

### Answer Correctness
Evaluates the overall correctness of the answer by comparing it with the ground truth. This metric considers both semantic similarity and factual overlap.

## Troubleshooting

**API Key Issues:**
- Ensure your API key is set correctly or entered in the sidebar
- For Groq: Get a free key at https://console.groq.com
- For OpenAI: Check that you have sufficient API credits

**Import Errors:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Try upgrading pip: `python -m pip install --upgrade pip`

**File Format Issues:**
- Ensure your file has all required columns
- For CSV files with lists, use proper JSON formatting for the contexts column

## Why Groq?

Groq offers:
- ⚡ **Fast inference** - Great for demos
- 🆓 **Free tier** - Perfect for testing
- 🤖 **Multiple models** - Llama 3.1, Mixtral, and more
- 📊 **Good performance** - Comparable to OpenAI for evaluation tasks

## Requirements

- Python 3.8 or higher
- Groq API key (free - get from https://console.groq.com) OR OpenAI API key
- Internet connection for API calls

## License

MIT License
