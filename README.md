# Shriyaa Sooklal: Personal Codex Agent

This is my trial project for Ubundi. It’s a context-aware chatbot that answers questions about me as a candidate, using my CV and supporting documents as its knowledge base.

## How it Works

-   Documents in the `data/` folder (CV, personal story, values, projects, etc.) are embedded with `sentence-transformers` and stored in a FAISS index.
-   When a user asks a question, the most relevant chunks are retrieved and passed to a GPT model along with a structured system prompt.
-   The chatbot answers in my voice, drawing on my personal documents and context.

## Tech Stack

-   **Python** (Streamlit for UI, FAISS for vector search, sentence-transformers for embeddings)
-   **OpenAI GPT-4o-mini** as the language model
-   Logs of all interactions saved in `artifacts/log.csv`

## API Key Handling

This project uses the OpenAI API.\
For security, the app does **not** include my key.

To use the demo: 1. Launch the app (`streamlit run app.py` locally, or open the deployed link).\
2. Paste your own OpenAI API key in the sidebar.\
- This ensures you can test with your own quota and settings.\
- It also avoids leaking keys in the repo.

In production, this could be adapted to: - Use environment variables (for secure team demos), or - Swap in an open-source LLM (for API-free deployments).

## Dataset Included

-   `CV.pdf`
-   `personal_story.txt`
-   `values_and_workstyle.txt`
-   `projects_proud_of.txt`

## Sample Questions

-   “What kind of engineer are you?”\
-   “What projects are you most proud of?”\
-   “What do you value in a team or company culture?”\
-   “How do you approach learning or debugging something new?”\
-   “Tell me about yourself.”

## Improvements with More Time

-   Add more “modes” (e.g. self-reflective, fast facts).
-   Add ability to update the dataset via upload.
-   Deploy with a secure backend API key for a zero-friction demo.
-   Experiment with multi-agent orchestration for richer responses.

## Show Your Thinking

Included in `artifacts/`: - Prompt files - Interaction logs - AI-assisted coding steps and commit messages

## My Repo Structure

```         
personal-codex-agent/
├── app.py                   # Streamlit UI
├── requirements.txt
├── prompts/
│   └── system_prompt.txt    
├── data/                    
│   └── README.md
├── scripts/
│   └── build_index.py       
├── artifacts/
│   └── log.csv             
└── README.md
```
