# RailCog: Benchmarking VLMs for Cognitive Railway Intrusion Perception

## Project Overview

RailCog is the first multimodal benchmark and open-source framework dedicated to cognitive railway intrusion perception, built on real-world surveillance scenes with cognition-driven, multi-dimensional instruction-level annotations (the CogRail dataset). It supports spatio-temporal reasoning, motion prediction, and threat assessment for objects of interest (OOIs) in railway environments. The project integrates visual question-answer annotations with expert-defined threat semantics and leverages instance synthesis to enhance data diversity while maintaining consistent label space across subsets. RailCog systematically evaluates state-of-the-art vision-language models (such as Qwen-VL and LLaVA) in railway scenarios, revealing their strengths and limitations in complex spatio-temporal reasoning. It also introduces the RAILGPT multi-task fine-tuning framework, which combines visual prompts, textual instructions, and specialized agents to optimize cognitive capabilities across position awareness (RailPos), motion prediction (RailMove), and threat analysis (RailThreat) tasks. After joint fine-tuning, RAILGPT achieves an 18.6% F1 improvement on the threat analysis task, demonstrating the effectiveness of structured multi-task learning in safety-critical scenarios and providing a complete benchmark toolchain for both research and engineering applications.

---

## 📊 Figures

### Figure 1: CogRail Dataset Construction Pipeline
[📄 Click here to download PDF](assets/dataset-pipeline.pdf)  
<p align="center">
  <object data="assets/dataset-pipeline.pdf" type="application/pdf" width="85%">
    <p>Your browser does not support embedded PDFs.  
    <a href="assets/dataset-pipeline.pdf">Click to download Figure 1</a>
    </p>
  </object>
</p>

---

### Figure 2: Threat Level Distribution & Object Composition
[📄 Click here to download PDF](assets/statistics.pdf)  
<p align="center">
  <object data="assets/statistics.pdf" type="application/pdf" width="85%">
    <p>Your browser does not support embedded PDFs.  
    <a href="assets/statistics.pdf">Click to download Figure 2</a>
    </p>
  </object>
</p>

---

### Figure 3: RAILGPT Multi-Task Learning Architecture
[📄 Click here to download PDF](assets/framework.pdf)  
<p align="center">
  <object data="assets/framework.pdf" type="application/pdf" width="85%">
    <p>Your browser does not support embedded PDFs.  
    <a href="assets/framework.pdf">Click to download Figure 3</a>
    </p>
  </object>
</p>

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)]()
[![License Code](https://img.shields.io/badge/License-Apache--2.0-green)]()
[![License Data](https://img.shields.io/badge/Data-CC%20BY--NC%204.0-lightgrey)]()

---

## ✨ Key Contributions (Highlights)
- **First CogRail Benchmark**: Integrates open-source surveillance scenarios with **cognition-driven question-answer annotations**, supporting spatio-temporal reasoning and intrusion risk prediction.  
- **Systematic Evaluation of Representative VLMs**: Reveals model strengths and weaknesses in cognitive railway scenarios.  
- **Multi-task Joint Fine-tuning (RAILGPT)**: Employs **visual prompts + textual prompts + dedicated agents** to significantly enhance accuracy and interpretability.  

---

## 📚 Benchmark & Task Definitions

### Three Core Tasks
- **RailPos (Spatial Awareness)**: Determine OOI location relative to railway infrastructure.  
- **RailMove (Motion Prediction)**: Predict threat level of movement.  
- **RailThreat (Threat Assessment)**: Integrate spatial + motion info to assess threat.  

---

## 🧰 Dataset (CogRail) Details

- **Sources**: MRSI, RailSem19 + LVIS for object synthesis  
- **Labels**: Category, Location, Motion, Threat + instruction-level QA  
- **Unified Label Space** across subsets for joint training  

---

## 🛠️ Technical Implementation

**RAILGPT** combines visual encoder, textual prompts, and specialized agents for position awareness, motion analysis, and final threat assessment.

| Model    | RailPos Acc | RailMove F1 | RailThreat F1 |
|----------|------------|-------------|---------------|
| Qwen-VL  | 82.3%      | 76.5%       | 68.2%         |
| LLaVA    | 78.1%      | 72.3%       | 63.7%         |
| RAILGPT  | **85.9%**  | **81.2%**   | **79.8%** (↑18.6%) |
