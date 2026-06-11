
# 🚀 2-Week AI Product Sprint: Fintech RegTech Copilot

**Objective:** Build a functional, full-stack Generative AI (RAG) Compliance Assistant to demonstrate robust, enterprise-ready AI capabilities on your CV.

**Tech Stack:** Python (FastAPI), React, Claude 3.5 Sonnet (Anthropic API), Kiro IDE, and ChromaDB.

---

## 🛠️ Sprint Pre-Requisites & Environment Setup

Before Day 1 begins, ensure your workspace is configured and your foundational accounts are ready.

* [ ] **Anthropic Developer Console Setup:** Create an account at [console.anthropic.com](https://console.anthropic.com/), generate a live API key, and fund it with $5–$10 of initial developer credit.
* [ ] **Download and Install Kiro:** Visit `kiro.dev` to download and install the standalone Kiro desktop application (or establish the Kiro CLI tool) for your operating system.
* [ ] **Environment Synchronization:** Open the standalone Kiro editor and import your existing VS Code extensions, preferences, and color themes into Kiro's AI-ready environment.
* [ ] **Workspace Initialization:** Create a local directory named `fintech-compliance-copilot`, open your terminal inside it, and run `kiro .` to open the folder inside your standalone agentic IDE.
* [ ] **Target Compliance Dataset Sourced:** Download the official **PCI DSS v4.0 Quick Reference Guide** or a condensed PDF summary from the PCI Security Standards Council document library. Save it inside a new `/data/raw/` subdirectory.

---

## 📅 Week 1: Requirements, System Architecture & The AI Engine

### Day 1–2: Product Requirements (PRD) & Spec-Driven Scoping

*Focus on defining strict boundary conditions and translating user intent into structured engineering inputs using Kiro's agentic spec workflow.*

* [ ] **Step 1.1: Create the Steering File (`product.md`)** Initialize a file named `product.md` in your root or `.kiro/steering/` directory to lock down product guardrails. Include your exact target user (Fintech PMs) and clear rules (e.g., "The model must explicitly cite clause numbers and output definite risk statuses: Compliant, Non-Compliant, or Warning").
* [ ] **Step 1.2: Trigger Kiro Spec Mode** Open Kiro's Agentic Chat or use its spec session tool. Input the following prompt to kickstart the system planning:
`"Architect a Python-based RAG application that parses PCI DSS v4.0 documents, builds semantic vector embeddings using Claude 3.5 Sonnet/OpenAI, and exposes an endpoint for product managers to run compliance queries with strict citations."`
* [ ] **Step 1.3: Review EARS Notation Output** Verify that Kiro successfully translates your prompt into structured acceptance criteria using **EARS** (Easy Approach to Requirements Syntax) format. Review, adjust, and approve the generated files.
* [ ] **Step 1.4: Finalize the Technical Specification (`spec.md`)** Ensure Kiro produces a complete `spec.md` detailing the data schema, ingestion pipeline, chunking layout, and FastAPI endpoint routes.

### Day 3–4: Document Processing, Chunking & Embeddings

*Build the ingestion engine that transforms dense regulatory text into numerical vectors that an LLM can parse contextually.*

* [ ] **Step 2.1: Establish the Data Ingestion Script** Instruct Kiro to generate a Python service (`ingestion.py`) utilizing libraries like `PyPDF2` or `pdfplumber` to extract clean text from your PCI DSS PDF.
* [ ] **Step 2.2: Implement the Chunking Strategy** Configure Kiro to implement a recursive character text splitter. Set an optimal chunk size (e.g., 500–800 characters) with a 10% overlap (50 characters) to avoid cutting off critical context between clauses.
* [ ] **Step 2.3: Generate Text Embeddings** Wire the system to send text chunks to the Anthropic/OpenAI embedding API, converting raw text paragraphs into mathematical vector arrays.
* [ ] **Step 2.4: Set Up the Vector Database** Initialize an embedded instance of `ChromaDB` (or a free cloud index on `Pinecone`). Test the ingestion pipeline by running a script that populates the database with your chunked compliance vectors.

### Day 5: Context Retrieval & Prompt Engineering (RAG Core)

*Connect user inquiries to your database to isolate and fetch the exact compliance clauses required before hitting the LLM.*

* [ ] **Step 3.1: Build Semantic Search Querying** Write a retrieval function that takes a plain-English user question (e.g., "Can we store CVV codes locally?"), converts it into an embedding vector, and performs a cosine-similarity search against your database to return the top 3 most relevant text chunks.
* [ ] **Step 3.2: Engineer the System Prompt & Guardrails** Design a rigorous system prompt instructing Claude 3.5 Sonnet to behave as an elite RegTech auditor. Enforce strict instructions: *"Answer the question using ONLY the provided text chunks. If the answer cannot be found in the context, explicitly state 'Clause not found in source documentation.' Never invent clause numbers."*
* [ ] **Step 3.3: Execute End-to-End CLI Testing** Run a test loop in your terminal. Ensure that typing a compliance query outputs a clean JSON response containing a plain-English answer, a risk classification, and the direct source text citation.

---

## 📅 Week 2: Full-Stack Integration, UX, Testing & CV Launch

### Day 6–8: Backend API & Frontend Dashboard Construction

*Expose your RAG engine via high-speed API routes and build an intuitive, professional dashboard UI.*

* [ ] **Step 4.1: Establish FastAPI Endpoints** Instruct Kiro to spin up a FastAPI server (`main.py`) exposing two primary endpoints:
1. `POST /api/query` (Accepts user questions, triggers the retrieval system, runs the prompt through Claude, and returns the audited response).
2. `GET /api/health` (System monitoring).


* [ ] **Step 4.2: Build the Frontend Layout (React / Retool)** Create a streamlined UI. If using React, let Kiro scaffold a single-page interface with a text entry box, a prominent "Run Compliance Audit" button, and a clean results area.
* [ ] **Step 4.3: Design Visual Status Indicators** Implement conditional styling on the front-end to render distinct UI blocks based on the AI's risk status output:
🟢 **Green Card:** Compliant
🟡 **Amber Card:** Warning / Conditions Apply
🔴 **Red Card:** Non-Compliant / Violation Detected
* [ ] **Step 4.4: Integrate Citation Callouts** Ensure the UI maps the source citations into clickable accordion drop-downs or footnote callouts so users can instantly verify the source document paragraphs.

### Day 9–11: Defending Against Hallucinations & Managing Edge Cases

*Polish the product by implementing guardrails, running error verification, and preparing for professional presentation.*

* [ ] **Step 5.1: Address Complex Queries & Ambiguity** Test the assistant with edge-case inputs (e.g., multi-part compliance questions, contradictory scenarios). Tune your prompt chains until the AI handles ambiguity gracefully without breaking character.
* [ ] **Step 5.2: Configure Automated Background Hooks** Leverage Kiro's background hooks feature. Set up a hook that automatically runs basic validation tests on your code and checks for latency variations whenever your system files change.
* [ ] **Step 5.3: Implement Graceful Error Handling** Ensure that network failures, API rate limits, or blank inputs result in clean UI error states instead of breaking the browser layout.

### Day 12–14: Launch Preparation, Portfolio Recording & CV Rewrite

*Package your creation into an undeniable asset and completely update your professional portfolio.*

* [ ] **Step 6.1: Record a High-Impact Video Walkthrough** Record a polished, 2-to-3 minute video demonstration using Loom. Structure your script like a Product Director:
1. *The Problem:* Fintech teams face massive velocity bottlenecks waiting for manual compliance reviews.
2. *The Solution:* Demonstrate your app instantly auditing a high-risk feature idea, flagging a payment tokenization violation, and surfacing direct PCI DSS citations.
3. *The Architecture:* Briefly highlight your RAG data pipeline and spec-driven agentic engineering workflow.


* [ ] **Step 6.2: Publish the Codebase** Push your clean, organized repository to GitHub. Write an institutional-grade `README.md` file featuring an architecture diagram, installation instructions, and the Loom video link.
* [ ] **Step 6.3: Update Your Professional CV** Add a dedicated, high-leverage entry to your resume showcasing your new AI technical fluency.

---

## 📝 CV Impact Blueprint: How to Frame This Project

Add this directly to your professional profile or dedicated "AI Initiatives" section to instantly grab the attention of executive stakeholders and engineering directors:

> **AI Product Initiative: GenAI RegTech Compliance Copilot (Production-Ready Prototype)** (2026)
> * Architected and deployed a zero-to-one Generative AI compliance assistant utilizing **Retrieval-Augmented Generation (RAG)** to accelerate feature onboarding loops for Fintech product engineering teams.
> * Formulated structured technical requirements, data models, and automated background chunking validation hooks within a spec-driven, agentic development framework (**Kiro IDE**).
> * Built a full-stack architecture leveraging **Python (FastAPI)** and **React**, integrating a vector database (**ChromaDB**) with **Claude 3.5 Sonnet** to execute semantic similarity search against dense regulatory text (**PCI DSS v4.0**).
> * Eliminated LLM hallucination risk through strict semantic grounding constraints and engineered automated citation engines, reducing the manual compliance verification cycle simulation from hours to sub-second responses.