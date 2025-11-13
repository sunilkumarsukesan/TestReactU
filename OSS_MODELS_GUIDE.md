# Using OSS Models with RAG Evaluation

This guide shows how to use open-source models with the RAG evaluation app.

## Option 1: Together AI (Recommended for OSS Models)

Together AI provides easy access to many open-source models with an OpenAI-compatible API.

### Setup:
1. Get API key from: https://api.together.xyz/
2. In the app sidebar:
   - Select: **"Custom OpenAI-Compatible"**
   - API Base URL: `https://api.together.xyz/v1`
   - Enter your Together AI API key
   - Model Name: Choose from below

### Popular Models:
- `mistralai/Mixtral-8x7B-Instruct-v0.1` - Great for evaluation tasks
- `meta-llama/Llama-3-70b-chat-hf` - High quality
- `meta-llama/Llama-3-8b-chat-hf` - Faster, cheaper
- `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO` - Excellent instruction following
- `Qwen/Qwen2-72B-Instruct` - Strong performance

## Option 2: OpenRouter

OpenRouter aggregates multiple providers with a single API.

### Setup:
1. Get API key from: https://openrouter.ai/
2. In the app sidebar:
   - Select: **"Custom OpenAI-Compatible"**
   - API Base URL: `https://openrouter.ai/api/v1`
   - Enter your OpenRouter API key
   - Model Name: e.g., `meta-llama/llama-3-70b-instruct`

## Option 3: Groq (Fastest Free Option)

Groq provides ultra-fast inference with free tier.

### Setup:
1. Get API key from: https://console.groq.com
2. In the app sidebar:
   - Select: **"Groq"**
   - Enter your Groq API key
   - Select model: **llama-3.3-70b-versatile** (recommended)

### Available Models:
- `llama-3.3-70b-versatile` - Latest, best quality
- `llama-3.1-8b-instant` - Fastest
- `mixtral-8x7b-32768` - Good for long contexts
- `gemma2-9b-it` - Efficient

## Option 4: Self-Hosted/Local Models

If you have a local model running with an OpenAI-compatible API:

### Setup:
1. Start your local model server (e.g., vLLM, Ollama with OpenAI compatibility, LocalAI)
2. In the app sidebar:
   - Select: **"Custom OpenAI-Compatible"**
   - API Base URL: Your local endpoint (e.g., `http://localhost:8000/v1`)
   - API Key: Your local API key (or any string if not required)
   - Model Name: Your model name as configured in your server

### Example with vLLM:
```bash
# Start vLLM server with OpenAI API
vllm serve meta-llama/Llama-3-8B-Instruct --api-key dummy-key
```

Then in the app:
- Base URL: `http://localhost:8000/v1`
- API Key: `dummy-key`
- Model: `meta-llama/Llama-3-8B-Instruct`

## Recommended Models for Evaluation

For RAG evaluation tasks, these models work well:

1. **Best Quality**: 
   - `meta-llama/Llama-3-70b-chat-hf` (Together AI)
   - `llama-3.3-70b-versatile` (Groq - Free!)

2. **Best Speed/Cost**:
   - `llama-3.1-8b-instant` (Groq - Free!)
   - `meta-llama/Llama-3-8b-chat-hf` (Together AI)

3. **Best for Long Context**:
   - `mixtral-8x7b-32768` (Groq - Free!)
   - `mistralai/Mixtral-8x7B-Instruct-v0.1` (Together AI)

## Cost Comparison

- **Groq**: FREE with rate limits (best for demos/testing)
- **Together AI**: Pay-per-use, generally cheaper than OpenAI
- **OpenRouter**: Variable pricing, often competitive
- **Self-hosted**: Infrastructure costs only, unlimited usage

## Troubleshooting

### "Model not found" error:
- Double-check the exact model name from the provider's documentation
- Some providers use different naming conventions

### API connection errors:
- Verify the base URL is correct
- Check if your API key is valid
- Ensure you have credits/quota remaining

### Slow evaluation:
- Try a smaller/faster model
- Use Groq for fastest inference
- Check your internet connection for cloud APIs
