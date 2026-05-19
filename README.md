<<<<<<< HEAD
# 🎓 xCARE - AuraTutor: Decision Support System

**xCARE** (eXplainable Continual Adaptation and Remediation Engine) is a Human-in-the-Loop Decision Support System (DSS) designed to manage Concept Drift adaptation in Intelligent Tutoring Systems (ITS). The **AuraTutor** ecosystem integrates an **xLSTM** architecture coupled with a dynamic **Continual-LoRA** router and an **Agentic RAG** engine (powered by Llama-3.3) constrained by Pydantic to generate hallucination-free pedagogical remediations.

This repository contains the complete source code and experimental setup validating the xCARE framework, provided for scientific reproducibility.

---

## 📂 Project Structure

* `teacher_dashboard.py`: Main interfaces for the educator dashboard developed with Streamlit.
* `student.py`: Learner interface designed to capture real-time interaction telemetry.
* `adaptation_engine.py`: The Continual-LoRA engine that dynamically adjusts the adaptation rank (r) based on the drift topology.
* `rag_engine.py`: The Agentic RAG module utilized to generate Explainable AI (XAI) pedagogical rationales.
* `evaluate_rag.py`: Evaluation script employing the LLM-as-a-Judge methodology to systematically audit the generated interventions.
* `knowledge_base.json`: The educational knowledge base mapping skills to targeted remediation exercises.

---

## 🚀 Installation and Setup

**1. Clone the repository and install dependencies**
```bash
git clone [https://github.com/Anonymous/xCARE-AuraTutor.git](https://github.com/Anonymous/xCARE-AuraTutor.git)
cd xCARE-AuraTutor
pip install torch peft langchain langchain-groq pydantic streamlit pandas

**2. Configure the API Key**
The Agentic RAG module requires a valid API key to query the LLM.
* On Windows (Command Prompt): `set GROQ_API_KEY=your_api_key_here`
* On macOS / Linux: `export GROQ_API_KEY="your_api_key_here"`

---

## 🖥️ Usage

**Launch the Educator Dashboard (AuraTutor DSS):**
```bash
streamlit run app.py

## 🔬 Scientific Reproducibility
This codebase accompanies the manuscript submission to the Expert Systems journal. By running this environment, reviewers and researchers can successfully reproduce the core empirical findings:

1. Elimination of Catastrophic Forgetting: Verification of the +0.00% Backward Transfer (BWT) via foundational weight freezing.

2. Rapid Algorithmic Adaptation: Predictive recovery across sudden, gradual, and recurring drift topologies using dynamic rank allocation.

3. Structural Hallucination Mitigation: Achieving a perfect structural adherence score (5.00/5.00) when executing the evaluate_rag.py auditing pipeline.
=======
# AuraTutor-xCARE
>>>>>>> b500077bf3ef900fb4af57d2e4819f19073118d6
