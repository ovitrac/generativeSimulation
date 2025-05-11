# 🌱 **What is Generative Simulation**?

**Generative Simulation** is an emerging paradigm that develops **advanced numerical simulations using elemental "bricks" accessible to text-generative AI models**. 

🧠 Tools like ChatGPT and its consorts excel at programming across many languages and grasping high-level, macroscopic concepts. However, they cannot seamlessly connect physics, chemistry, biology, and mathematics to solve real-world scientific and engineering problems.

**Generative Simulation** provides modular bricks—representations of physical or conceptual building blocks—that Large Language Models (LLMs) can understand and manipulate to bridge this gap. These bricks can be designed with AI assistance, but the overarching logic, scientific insight, and problem-specific nuance remain in human hands.

Once the bricks form a structured language, the subsequent stages of model development, simulation assembly, or code generation can be delegated back to the AI and iteratively refined under human supervision.

> 🧩 These bricks provide a clear context that LLMs can follow within their existing context window (from 8K to 128K tokens). **Simulations** and **scenarios** can be produced from prompts including specific instructions. Clear examples, reusable classes, and operators overcome the **limitations of the current window size**.
>
> 🚧 A prototype demonstrating LFCL orchestration is under development and is drafted in this document.



![Illustrative examples from Vitrac work](https://raw.githubusercontent.com/ovitrac/generativeSimulation/main/assets/page2.png)

<small>🎨 Credits: Olivier Vitrac</small>

---

## 📦 Some Showcases

🔧 The **Generative Simulation Ecosystem** already includes powerful computational/multiscale kernels for:
- ⚙️ Mechanics (e.g., particle-based models, forcefields): large deformations, mechanical rupture, adhesion/melting
- 💧 Mass Transfer (e.g., Fickian PDEs, molecular diffusivity and sorption estimators)
- 🔥 Chemical Reactivity (e.g., ODE systems, combinatorial and graph-based networks)

> 📚 Each kernel is enriched with embedded domain-specific databases: 🧪 Forcefields, 🧊 Thermodynamic data, 🧬 Transport properties, ⚗️ Reactivity schemes and reaction rate constants, ⌬ chemical databases (e.g., PubChem), ☢️ toxicological databases, ⚖️ legal databases. 

🧠 These components are modular and readable by LLMs, enabling simulation-driven reasoning, scenario generation, and iterative refinement through natural language prompts.

---

### 🍕 Example 1 | **Pizza³** – A Multiscale Python Toolkit

**Pizza3** simulates the mechanics of soft matter using **LAMMPS**, with fully reusable Python objects that AI can understand and extend. This modular approach simplifies multiscale physics integration while abstracting away unnecessary technicalities for HPC/cloud deployment.

- **💻 Computational resources**: ++++  
- **🧠 Complexity**: +++  
- **🚀 Typical deployment**: HPC, GPU, AWS  
- **🔗 Source**: [Pizza3 on GitHub](https://github.com/ovitrac/Pizza3)

---

### 🍽️ Example 2 | **SFPPy** – Compliance & Risk Assessment for Food Contact Materials

**SFPPy** is a Python framework that accelerates the evaluation of material safety in contact with food. It connects to regulatory databases, computes migration properties from molecular structures, and simulates mass transfer across packaging layers.

Already recognized by EU, US, and Chinese agencies, SFPPy integrates:
- Mass transfer models
- AI-generated scenarios
- Jupyter-based documentation and reports

- **💻 Computational resources**: +  
- **🧠 Complexity**: +  
- **🖥️ Deployment**: Colab, Jupyter Server  
- **📘 Usage**: Notebooks for modeling, tracking, and AI-assisted reporting  
- **🔗 Source**: [SFPPy on GitHub](https://github.com/ovitrac/SFPPy)

---

### 🌐 Example 3 | **SFPPylite** – SFPPy in Your Browser

**SFPPylite** is a WebAssembly-based version of SFPPy running entirely in your browser—no server, no installation required.

Perfect for:
- 🏫 Classroom training
- 🧪 SME simulations
- 📝 Regulatory collaboration (copy/paste into chatbots or notebooks)

- **💻 Computational resources**: +  
- **🧠 Complexity**: +  
- **🖥️ Deployment**: Browser (Chrome, Firefox)  
- **🔗 Source**: [SFPPylite on GitHub](https://github.com/ovitrac/SFPPylite)

---

### 🔬 Example 4 | **Radigen** – Kernel for Radical Oxidation Mechanisms

**Radigen** simulates radical oxidation in **edible oils**, **biofuels**, and **polymer degradation**. It uses reactive functions, class inheritance, and mechanistic logic to build large-scale reaction networks with thousands of time-dependent species.

Composable Python bricks:
- `mixture` defines species and reactions
- `mixtureKinetics` solves the ODE system
- `lumped` aggregates similar species
- `TKO2cycle` controls T/O₂ cycles

> 💬 Prompt example:  
> “Simulate oxidation of linoleic acid for 3 days at 160°C and partial renewal with fresh oil.”

- **💻 Computational resources**: +/++  
- **🧠 Complexity**: +/++  
- **🖥️ Deployment**: Scripts, notebooks, LLM-assisted workflows  
- **🔬 Usage**: Food, pharma, cosmetics, polymer stability  
- **🔗 Source**: [Radigen on GitHub](https://github.com/ovitrac/radigen)

---

# 🧪 Part 1 | **Language-First Computational Lab (LFCL) Road Map**

**Generative Simulation** enables a **Language-First Computational Lab (LFCL)**—a new paradigm where **LLMs orchestrate numerical simulations** using modular physics/chemistry/biology kernels with human-guided workflows.

## 1.1 | 🌐 Core Philosophy

> In a language-first lab, *natural language* becomes the **primary interface** for composing, querying, refining, and explaining scientific models.

LLMs:
- 🛠️ Do **not** hold full models
- 🤖 **Control, audit**, and **refine** simulations via external kernels
- 🧠 Provide bidirectional reasoning with humans

---

## 1.2 | 🧩 Key Components

### (A) 🧪 Kernel Ecosystem
Modular simulation kernels accessible via APIs, ideally stateless:

- Input: structured physical quantities
- Output: machine-readable + human-readable

Examples:
- `mass_transfer_kernel(diffusivity, geometry, time)`  
- `radigen_kernel(reaction_network, T, dt)`

---

### (B) 🤖 Language Layer (LLM Agent)

LLM capabilities include:
- Interpretation (e.g., "what if T ↑ 10°C?")
- Scripting DSL or Python code
- Fitting, analysis, assumption validation
- Detecting uncertainty or ill-posed logic

---

### (C) 🧠 Context Memory

Buffers tailored to **8K–128K token windows**:

- Tracks assumptions, results
- Summarizes intermediate states
- Enables iterative refinement

---

### (D) 👩‍🔬 Human-in-the-Loop Interface

Dashboards or notebooks to:
- Override model logic
- Display results/uncertainties
- Compare and version simulations

---

## 1.3 | 🔄 Data & Model Flow

```mermaid
graph TD
A[User Input<br>Natural Language Prompt]
A -->|interpreted| B[<b>LLM Agent</b]
B -->|calls| C[Simulation Kernel]
C --> D[Output: Structured + Natural Language]
D -->|interpreted| B
B --> E[Notebook / Dashboard]
E --> A
````

---

## 1.4 | 🧑‍💼 Role of the Human Architect

Humans (engineers, scientists, technicians) define and guard simulation logic:

* 🎯 Define kernel schemas
* 🔍 Validate logic
* 📐 Audit assumptions
* 🔁 Design prompt workflows

The human becomes the **workflow architect**.

---

## 1.5 | 🔧 Application Scenarios

The applications are numerous. Current kernels can already provide answers in several domains.

| Domain                              | Example Workflow                                             |
| ----------------------------------- | ------------------------------------------------------------ |
| **Polymer Migration**               | “Estimate PET monomers migration after 10 days at 60°C”      |
| **Oxidation Kinetics**              | “Which species peaks first during FAME oxidation at 90°C?”   |
| **Thermomechanics**                 | “Simulate stress in packaging during sterilization”          |
| **Circular Design**                 | “Propose recycled polymer blends with acceptable safety and strength” |
| **Haptic and Mouthfeel perception** | “Simulate the texture perception of a mixture of ….”         |

---

## 1.6 🪄 Beyond the Language Window

To scale beyond context limits:

* Use external **caches** or memory modules
* Break work into **episodes** with symbolic anchors
* Export results as **summarized memory tokens**

---

# 🧰 Part 2 | **LFCL Architecture (Sketch)**

> This architecture supports real-time orchestration of physical/chemical/biological kernels by LLMs using token-efficient prompts.

---

## 2.1 | 🔄 Core Conceptual Layers

```mermaid
graph TD
A["👩‍🔬 USER<br>(Scientist / Engineer)"]
A --> B["🗣️ Natural Language Prompt"]
B --> C["🧠 <b>LLM Interpreter</b><br>(Language Layer)"]
C --> D["🧾 Function Call / Code Generation"]
D --> E["⚙️ Simulation Kernels<br>(Physics / Chemistry / Biology)"]
E --> F["📊 Results<br>(Structured Data)"]
F --> G["🔁 <b>LLM Interpreter</b><br>(Backloop)"]
G --> H["📝 Narrative Summary + Scientific Explanation"]

```

---

## 2.2 🗂️ File Structure (Prototype)

```bash
LFCL/
├── main.py                     # Entry point
├── agent.py                    # LLM orchestrator
├── memory.py                   # Token-aware memory manager
├── interface.py                # CLI, notebook or Streamlit
├── kernels/
│   ├── radigen_interface.py    # Oxidation modeling API
│   ├── sfppy_interface.py      # Migration modeling API
│   └── core_utils.py           # Shared utilities
├── prompts/
│   └── templates.md            # Prompt interpreters
├── config/
│   └── kernel_schema.yaml      # I/O standards
└── logs/
    └── session_001.json
```

---

## 2.3 | 🔬 `radigen_interface.py`

```python
def run_oxidation_simulation(species_list, T=80, duration=("10", "d")):
    from radigen.radigen3.mixtureKinetics import mixtureKinetics
    mk = mixtureKinetics.from_species(species_list)
    mk.solve(time=duration, T=T)
    return mk.get_dataframe()
```

Minimal API:

* `from_species(...)`
* `.solve(...)`
* `.get_dataframe()`

---

## 2.4 | 🧠 `agent.py`

```python
def handle_prompt(prompt, context):
    if "oxidation" in prompt and "FAME" in prompt:
        species = ["monoAllylicCOOH", "oxygen"]
        T = 90
        time = (10, "h")
        from kernels.radigen_interface import run_oxidation_simulation
        df = run_oxidation_simulation(species, T, time)
        return summarize_results(df, species)
```

---

## 2.5 | 🧠 `memory.py`

```python
class ContextMemory:
    def __init__(self, max_tokens=8000):
        self.entries = []
        self.max_tokens = max_tokens

    def add(self, text):
        self.entries.append(text)
        self._trim()

    def _trim(self):
        ...
```

Tracks:

* Results
* Assumptions
* Code chunks

---

## 2.6 | 💻 Notebook/Streamlit Interface

```python
import streamlit as st
from agent import handle_prompt

prompt = st.text_area("Ask your lab:", "Simulate oxidation of FAME at 90°C")
if st.button("Run"):
    result = handle_prompt(prompt, context={})
    st.write(result)
```

---

## 2.7 | 🧾 Prompt Template (`templates.md`)

```markdown
# Template: Oxidation Modeling

**User Prompt**
"Can you simulate the oxidation of methyl linoleate at 80°C for 5 days?"

**Parsed Actions**
- domain = oxidation
- model = radigen
- species = ["monoAllylicCOOH", "oxygen"]
- time = (5, "d")
- T = 80°C

**Kernel Call**
run_oxidation_simulation(...)

**Summary**
Simulated methyl linoleate oxidation. Products include L1OOH, ROO•. See plots.
```

---

📌 **Want to contribute?** Reach out via GitHub issues or discussions to propose new kernels, interface improvements, or educational use cases!



[olivier.vitrac@gmail.com]([olivier.vitrac@gmail.com]) | May 2025
