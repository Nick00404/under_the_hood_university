# 07 Prompt Engineering

- [01 intro to prompting zero one few shot](./01_intro_to_prompting_zero_one_few_shot.ipynb)  
- [02 chain of thought and reasoning prompts](./02_chain_of_thought_and_reasoning_prompts.ipynb)  
- [03 prompt tuning vs in context learning](./03_prompt_tuning_vs_in_context_learning.ipynb)  
- [04 function calling and tool use prompts](./04_function_calling_and_tool_use_prompts.ipynb)  
- [05 react agents and scratchpad prompting](./05_react_agents_and_scratchpad_prompting.ipynb)  
- [06 self improving prompts and auto eval loops](./06_self_improving_prompts_and_auto_eval_loops.ipynb)  
- [07 prompt injection defense and safety](./07_prompt_injection_defense_and_safety.ipynb)  
- [08 lab chain of thought prompt eval suite](./08_lab_chain_of_thought_prompt_eval_suite.ipynb)  
- [09 lab prompt injection red teaming](./09_lab_prompt_injection_red_teaming.ipynb)

---

## 🧠 **Prompt Engineering – Structured Index**

---

### 🧠 **01. Intro to Prompting: Zero, One, and Few-Shot**

#### 📌 **Subtopics:**
- **What is Prompting?**
  - Role of prompts in LLM interaction
- **Zero-shot vs Few-shot**
  - Instruction-only vs example-based prompting
- **Prompt Structure Patterns**
  - Direct instructions, Q&A, completion formats
- **Example:** Comparing zero-shot vs few-shot performance on sentiment classification

---

### 🧠 **02. Chain of Thought and Reasoning Prompts**

#### 📌 **Subtopics:**
- **Why Chain of Thought (CoT)?**
  - Improving reasoning by forcing intermediate steps
- **When CoT Works (and Fails)**
  - Arithmetic, logic puzzles, multi-hop QA
- **Variants: Tree of Thought, Program-Aided CoT**
- **Example:** Generating CoT prompts for math word problems and comparing token efficiency

---

### 🧠 **03. Prompt Tuning vs In-Context Learning**

#### 📌 **Subtopics:**
- **Prompt Tuning**  
  - Soft prompts and prefix tuning via optimization
- **In-Context Learning (ICL)**  
  - Runtime learning without parameter updates
- **Tradeoffs: Tuning vs Prompt Crafting**
- **Example:** Train a soft prompt for classification vs few-shot prompt on the same task

---

### 🧠 **04. Function Calling and Tool Use Prompts**

#### 📌 **Subtopics:**
- **Calling External Functions via Prompts**
  - Structured JSON outputs, OpenAI function schema
- **Toolformer and API-enabled Prompting**
  - Instructing the model to pick tools for tasks
- **Example:** Prompting GPT to call a calculator and a weather API in sequence

---

### 🧠 **05. ReAct Agents and Scratchpad Prompting**

#### 📌 **Subtopics:**
- **Reasoning + Acting (ReAct) Pattern**
  - Interleaving thinking steps and tool calls
- **Scratchpad Design**
  - Inserting memory/state within the prompt
- **Planning with External Memory**
- **Example:** Multi-step QA using a ReAct agent with search + calculator tools

---

### 🧠 **06. Self-Improving Prompts and Auto Eval Loops**

#### 📌 **Subtopics:**
- **Meta-Prompting**
  - Prompts that generate and improve other prompts
- **Evaluation + Selection Loops**
  - Prompt self-play, autoeval with scoring heuristics
- **Emerging Systems**
  - Promptbreeder, Prompt Arena, Prompt2Prompt
- **Example:** An automatic pipeline that evolves prompts for better summarization

---

### 🧠 **07. Prompt Injection Defense and Safety**

#### 📌 **Subtopics:**
- **Types of Injection Attacks**
  - Instruction override, jailbreak, adversarial inputs
- **Mitigation Strategies**
  - Input sanitation, role-based templates, output filtering
- **Security in Prompt-based APIs**
- **Example:** Simulating injection attempts and applying defense patterns

---

### 🧪 **08. Lab: Chain of Thought Prompt Evaluation Suite**

#### 📌 **Subtopics:**
- **Testing Framework**
  - Eval metrics for reasoning depth and step accuracy
- **Prompt Variants**
  - Systematic testing across templates
- **Visualization**
  - Token attribution, reasoning trace diagrams
- **Example:** Benchmarked CoT prompts on GSM8K with explanation heatmaps

---

### 🧪 **09. Lab: Prompt Injection Red Teaming**

#### 📌 **Subtopics:**
- **Injection Scenarios**
  - System override, “ignore above” tricks, Unicode exploits
- **Red Teaming Techniques**
  - Auto-gen of adversarial prompts, fuzz testing
- **Evaluation**
  - Success rate, robustness, and mitigation effectiveness
- **Example:** Build a red team simulator to stress test a function-calling assistant

---
