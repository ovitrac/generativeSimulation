# 🌱 **What is Generative Simulation**?

---

**Generative Simulation** is an emerging paradigm that develops **advanced numerical simulations using elemental "bricks" accessible to text-generative AI models**. 

🧠 Tools like ChatGPT and its consorts excel at programming across many languages and grasping high-level, macroscopic concepts. However, they cannot seamlessly connect physics, chemistry, biology, and mathematics to solve real-world scientific and engineering problems.

**Generative Simulation** provides modular bricks—representations of physical or conceptual building blocks—that Large Language Models (LLMs) can understand and manipulate to bridge this gap. These bricks can be designed with AI assistance, but the overarching logic, scientific insight, and problem-specific nuance remain in human hands.

Once the bricks form a structured language, the subsequent stages of model development, simulation assembly, or code generation can be delegated back to the AI and iteratively refined under human supervision.

> 🧩 These bricks provide a clear context that LLMs can follow within their existing context window (from 8K to 128K tokens). **Simulations** and **scenarios** can be produced from prompts including **specific instructions** and **experimental data** (*e.g.*, chromatograms). Clear examples, reusable classes, and operators overcome the **limitations of the current window size**.
>
> 🚧 A prototype demonstrating LFCL orchestration is under development and is drafted in this document.



![Illustrative examples from Vitrac work](https://raw.githubusercontent.com/ovitrac/generativeSimulation/main/assets/page2.png)

<small>🎨 Credits: Olivier Vitrac</small>



---

## Table of Contents 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

  - [📦 **Some Showcases**](#-some-showcases)
    - [🍕 Example 1 | **Pizza³** – A Multiscale Python Toolkit](#-example-1--pizza%C2%B3--a-multiscale-python-toolkit)
    - [🍽️ Example 2 | **SFPPy** – Compliance & Risk Assessment for Food Contact Materials](#-example-2--sfppy--compliance--risk-assessment-for-food-contact-materials)
    - [🌐 Example 3 | **SFPPylite** – SFPPy in Your Browser](#-example-3--sfppylite--sfppy-in-your-browser)
    - [🧪⚛️ Example 4 | **Radigen** – Kernel for Radical Oxidation Mechanisms](#-example-4--radigen--kernel-for-radical-oxidation-mechanisms)
    - [📡🧬 Example 5 | **sig2dna** – Symbolic Fingerprinting of Analytical Signals](#-example-5--sig2dna--symbolic-fingerprinting-of-analytical-signals)
- [🧪 Part 1 | **Language-First Computational Lab (LFCL) Road Map**](#-part-1--language-first-computational-lab-lfcl-road-map)
  - [1.1 | 🌐 Core Philosophy](#11---core-philosophy)
  - [1.2 | 🧩 Key Components](#12---key-components)
    - [(A) 🧪 Kernel Ecosystem](#a--kernel-ecosystem)
    - [(B) 🤖 Language Layer (LLM Agent)](#b--language-layer-llm-agent)
    - [(C) 🧠 Context Memory](#c--context-memory)
    - [(D) 👩‍🔬 Human-in-the-Loop Interface](#d--human-in-the-loop-interface)
  - [1.3 | 🔄 Data & Model Flow](#13---data--model-flow)
  - [1.4 | 🧑‍💼 Role of the Human Architect](#14---role-of-the-human-architect)
  - [1.5 | 🔧 Application Scenarios](#15---application-scenarios)
  - [1.6 | 🪄 Beyond the Language Window](#16---beyond-the-language-window)
  - [1.7 | 🌈⃤ Illustrations](#17--%E2%83%A4-illustrations)
- [🧰 Part 2 | **LFCL Architecture (Sketch)**](#-part-2--lfcl-architecture-sketch)
  - [2.1 | 🔄 Core Conceptual Layers](#21---core-conceptual-layers)
  - [2.2 | 🗂️ File Structure (Prototype)](#22---file-structure-prototype)
  - [2.3 | 🔬 `radigen_interface.py`](#23---radigen_interfacepy)
  - [2.4 | 🧠 `agent.py`](#24---agentpy)
  - [2.5 | 🧠 `memory.py`](#25---memorypy)
  - [2.6 | 💻 Notebook/Streamlit Interface](#26---notebookstreamlit-interface)
  - [2.7 | 🧾 Prompt Template (`templates.md`)](#27---prompt-template-templatesmd)
- [🧩🌱 Part 3 | Building Specialized Kernels](#-part-3--building-specialized-kernels)
  - [3.1 | 💼 What Are Specialized Kernels ?](#31---what-are-specialized-kernels-)
  - [3.2 | ✅ Kernel Requirements](#32---kernel-requirements)
  - [3.3 | 🍏⏩🍎 SFPPy as a Kernel Example](#33---sfppy-as-a-kernel-example)
- [👥👉 Want to contribute?](#-want-to-contribute)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

---

## 📦 **Some Showcases**

🔧 The **Generative Simulation Ecosystem** already includes powerful computational and multiscale kernels for:

- ⚙️ **Mechanics** – particle-based models and forcefields (LAMMPS-compatible): large deformations, rupture, adhesion/melting

- 💧 **Mass Transfer** – diffusion–reaction systems (Fickian PDEs, sorption models, molecular diffusivity estimators)

- 🔥 **Chemical Reactivity** – ODE networks, combinatorial pathway generators, graph-based transformations

- 📡 **Chemical Signal Encoding and Analysis** – symbolic transformation of 1D analytical signals (GC-MS, NMR, FTIR…)

  

> 📚 Each kernel is enriched with embedded **domain-specific knowledge** and scientific databases:
>
> -  🧪 **Forcefields** — atomistic, coarse-grained (DPD, SDPD, SPH)
> -  🧊 **Thermodynamic Data** — solubilities, mixing energies, phase transitions
> -  🧬 **Transport Properties** — diffusivity, dissolution, activation energy models (Arrhenius, free-volume)
> -  ⚗️ **Reactivity Schemes** — reaction networks, rate laws, temperature effects
> -  ⌬ **Chemical Databases** — PubChem integration, CAS handling, identifiers
> -  ⛓️ **Material Databases** — polymer properties (e.g., $T_g$, density, crystallinity)
> -  ☢️ **Toxicology Tools** — ToxTree predictions, hazard flags
> -  ⚖️ **Regulatory Frameworks** — legal thresholds (🇪🇺 EU, 🇺🇸 US, 🇨🇳 China)
> -  🫆 **Chemical Fingerprints** — symbolic encoding of GC-MS, FTIR, RAMAN, $^1$H-NMR
>
> 💡 Many models also integrate **practical factors across the value chain**:
>
> -  ♻️ Recycling & decontamination processes for plastics and cellulosic materials
> -  🧃🥡 Packaging types, geometries, and market statistics
> -  🥗🍛 Food matrix composition and reactivity
> -  ♨️🌡️❄️ Food transformation (e.g., deep-frying, pasteurization) and storage

🧠 All components are designed as **modular Python objects readable by LLMs**, enabling **simulation-driven reasoning**, **scenario exploration**, and **code generation** through natural language prompts.



**Examples Overview**

[![🍕 Pizza³](https://img.shields.io/badge/🍕_Pizza³-mechanics_/_forcefields-8e44ad?style=for-the-badge&labelColor=2c2c2c)](https://github.com/ovitrac/Pizza3)
[![🍽️ SFPPy](https://img.shields.io/badge/🍽️_SFPPy-food_safety_/_migration-2e7d32?style=for-the-badge&labelColor=1b1b1b)](https://github.com/ovitrac/SFPPy)
[![🌐 SFPPylite](https://img.shields.io/badge/🌐_SFPPylite-browser_WASM_/_compliance-0288d1?style=for-the-badge&labelColor=1c1c1c)](https://github.com/ovitrac/SFPPylite)
[![🧪⚛️ Radigen](https://img.shields.io/badge/🧪⚛️_Radigen-radical_oxidation_/_kinetics-e65100?style=for-the-badge&labelColor=2a2a2a)](https://github.com/ovitrac/radigen)
[![📡🧬 sig2dna](https://img.shields.io/badge/📡🧬_sig2dna-symbolic_signals_/_entropy_/_alignment-6a1b9a?style=for-the-badge&labelColor=1e1e1e)](https://github.com/ovitrac/sig2dna)




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

- **💻 Computational resources**: +/++  
- **🧠 Complexity**: ++  
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
- **🧠 Complexity**: ++  
- **🖥️ Deployment**: Browser (Chrome, Firefox)  
- **🔗 Source**: [SFPPylite on GitHub](https://github.com/ovitrac/SFPPylite)

---

### 🧪⚛️ Example 4 | **Radigen** – Kernel for Radical Oxidation Mechanisms

**Radigen** simulates radical oxidation in **edible oils**, **biofuels**, and **polymer degradation**. It uses reactive functions, class inheritance, and mechanistic logic to build large-scale reaction networks with thousands of time-dependent species.

Composable Python bricks:
- `mixture` defines species and reactions
- `mixtureKinetics` solves the ODE system
- `lumped` aggregates similar species
- `TKO2cycle` controls T/O₂ cycles

> 💬 Prompt example:  
> “Simulate oxidation of linoleic acid for 3 days at 160°C and partial renewal with fresh oil.”

- **💻 Computational resources**: +  
- **🧠 Complexity**: +/++  
- **🖥️ Deployment**: Scripts, notebooks, LLM-assisted workflows  
- **🔬 Usage**: Food, pharma, cosmetics, polymer stability  
- **🔗 Source**: [Radigen on GitHub](https://github.com/ovitrac/radigen)

---

### 📡🧬 Example 5 | **sig2dna** – Symbolic Fingerprinting of Analytical Signals

**sig2dna** transforms complex analytical signals—like **GC-MS**, **NMR**, or **FTIR** outputs—into compact, symbolic **DNA-like sequences** using a multi-scale wavelet encoding approach. These sequences enable **motif recognition**, **signal alignment**, and **unsupervised classification**, unlocking new capabilities for AI-assisted analysis of real-world chemical data.

This symbolic transformation compresses large signals (>95%) while preserving morphological detail. Once encoded, symbolic distances (e.g., **Excess Entropy**, **Jaccard**, **Levenshtein**, **Jensen-Shannon**) allow clustering, outlier detection, and **blind source separation**, even for overlapping or noisy spectra.

**sig2dna** is especially suited for:

- ♻️ NIAS (non-intentionally added substances) fingerprinting in recycled materials
- 🧪 Quality control in chemical, food, or cosmetic analysis
- 🤖 LLM-based workflows that search or cluster chemical patterns from symbolic codes

> 💬 Prompt example:
>  “Classify this mixture of GC-MS signals using entropy distance and extract all motifs matching `YAZB`.”

- **💻 Computational resources**: +

- **🧠 Complexity**: ++

- **🖥️ Deployment**: Python scripts, Colab, Jupyter

- **📊 Usage**: Signal classification, clustering, fingerprinting, AI-assisted interpretation

- **🔗 Source**: [sig2dna on GitHub](https://github.com/ovitrac/sig2dna)

  


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

## 1.6 | 🪄 Beyond the Language Window

To scale beyond context limits:

* Use external **caches** or memory modules
* Break work into **episodes** with symbolic anchors
* Export results as **summarized memory tokens**

---

## 1.7 | 🌈⃤ Illustrations

**The LFCL architecture enables an engineer or scientist to delegate complex scientific questions to an LLM-powered orchestration system.**
 Upon receiving a question **Q**, the **LLM Input-Agent** devises a strategy **S** that decomposes Q into sub-questions (**q₀, q₁, q₂, ...**), each routed to an appropriate computational process:

- **q₀**: interpreted through semantic knowledge
- **q₁, q₂, ...**: dispatched to **low-latency** or **intensive specialized kernels**, drawing on numerical databases
   The answers **a₀, a₁, a₂, ...** are consolidated by the **LLM Output-Agent**, which builds a structured response and report.

![LFCL proecess](https://raw.githubusercontent.com/ovitrac/generativeSimulation/main/assets/LFCLprocess.jpg)

<small>🎨 Credits: Olivier Vitrac</small>

>This architecture ensures reasoning and simulation are balanced, reducing delays and computational cost while preserving scientific rigor and traceability.


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

## 2.2 | 🗂️ File Structure (Prototype)

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



# 🧩🌱 Part 3 | Building Specialized Kernels



## 3.1 | 💼 What Are Specialized Kernels ?

**Specialized kernels** are domain-specific computational modules that extend the reasoning capabilities of language models into the realm of **quantitative science and engineering**. They are the **physical, chemical, or biological “brains”** of the Language-First Computational Lab (LFCL)—enabling LLMs to **compute, not just communicate**.

Unlike general-purpose LLMs, which rely on linguistic priors, these kernels:

- Accept **numerical inputs** (e.g., lab data, physical parameters, boundary conditions)
- Deliver **quantitative outputs** (e.g., reaction kinetics, migration rates, stress/strain curves)
- Provide **high precision, traceability, and interpretability** far beyond generic AI reasoning

🧠 Through the **LFCL interface**, these kernels are made accessible to LLMs via **semantic hooks and structured prompts**—allowing agents to reason over data, not just text.

![LFCL-based kernels](https://raw.githubusercontent.com/ovitrac/generativeSimulation/main/assets/LFCLkernels.jpg)

<small>🎨 Credits: Olivier Vitrac</small>

---

## 3.2 | ✅ Kernel Requirements

---

> ### ⚙️ **Key Requirements for Kernel Design**
>
> - Each kernel must implement a **LFCL-compatible interface**, ideally in **Python**, to support both LLM integration and human usage.
> - Kernel development should be **driven by existing validated scientific methods**—those supported by peer-reviewed literature, regulatory bodies, or industrial standards.
> - Clear **usage documentation** is mandatory, including:
>   - Valid domains of application
>   - Acceptable input/output ranges
>   - Explicit warnings for extrapolation or domain gaps
>   - Associated **uncertainty quantification**
> - The kernel type (⚡ low-latency vs 🧮 intensive) should match:
>   - The trustworthiness of the method
>   - The urgency or tier of the decision-making process
>   - The available computational resources
> - Wherever possible, **multitiered or progressive strategies** should be used—starting with fast approximations and refining only where necessary.

---

> ### 🔄 **On Interdisciplinarity and Reusability**
>
> - Kernels are most powerful when **combined across domains** (e.g., mass transfer + toxicology, or diffusion + reaction mechanics).
> - LFCL supports both **rough estimators** and **high-fidelity solvers** (e.g., particle-based, mesh-based, Monte Carlo, or thermodynamic integration).
> - **Registries** should list:
>   - Method provenance (e.g., peer-reviewed, regulatory, internal R&D)
>   - Maintenance status and contributors
>   - Example use cases and known limitations
> - Open repositories are encouraged to foster **independent validation**, reusability, and collective improvement beyond the LLM ecosystem.

---

> ### 🔁 **From Static Tool to Dynamic Reasoner**
>
> LLM agents can chain together **past requests and kernel invocations** to form **new composite simulations**. This enables deeper scenario construction and fosters creative reuses of existing scientific bricks.
>
> In essence, LFCL provides **augmented scientific intelligence** for both expert and non-expert users—**amplifying exploration and guiding attention toward meaningful results**. It acts as a catalyst for more robust reasoning, faster hypothesis testing, and greater transparency.


---

## 3.3 | 🍏⏩🍎 SFPPy as a Kernel Example

**SFPPy** is a fully integrated kernel designed for **food contact compliance and risk assessment**. It combines:

- A robust **mass transfer solver**
- **Molecular and toxicological databases**
- **Structure-based migration models**
- A comprehensive **documentation layer**
- Strong alignment with **EU, US, and Chinese regulatory frameworks**



![SFPPY as a showcase](https://raw.githubusercontent.com/ovitrac/generativeSimulation/main/assets/SFPPy.jpg)

<small>🎨 Credits: Olivier Vitrac</small>



As a kernel in the **LFCL ecosystem**, SFPPy empowers LLMs to interpret and solve **complex regulatory and technological queries** in natural language—spanning:

- Material design & formulation
- Food type & processing conditions
- Packaging structure & multi-layer barriers
- Migration limits & consumer exposure scenarios

⏱️ What would traditionally take **hours of expert interpretation** can now be framed, explored, and **drafted in seconds**, with results that are explainable, traceable, and reproducible.

> Example query:
>  **"Can you demonstrate that this recycled multilayer material complies with food contact regulations for fatty foods in the EU market after retort processing?"**

Thanks to its structured design and regulatory foundations, SFPPy can simulate such scenarios directly or support them with partial answers, identifying gaps or missing inputs.



---


# 👥👉 Want to contribute?


📌 **Want to contribute?** Reach out via GitHub issues or discussions to propose new kernels, interface improvements, or industrial/educational use cases!


![call for contributions](https://raw.githubusercontent.com/ovitrac/generativeSimulation/main/assets/GSjoin.jpg)

<small>🎨 Credits: Olivier Vitrac</small>


[olivier.vitrac@gmail.com]([olivier.vitrac@gmail.com]) | May 2025
