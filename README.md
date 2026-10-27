# TRACE@VPN: Agentic VPN Traffic Identification

**TRACE@VPN** is a research framework for identifying encrypted VPN and non-VPN network traffic using time-series learning and evidence-driven decision-making.

The framework uses a PatchTST-based model to analyze packet metadata without inspecting encrypted payloads. An agentic layer further evaluates the evidence to support transparent and reliable classification.

## Key Features

- **Short-context analysis:** Identifies traffic using five-second observation windows.
- **Deep learning:** Uses PatchTST for VPN and non-VPN classification.
- **Enhanced features:** Incorporates packet timing, packet-size changes, and traffic direction.
- **Evidence-driven analysis:** Combines predictions with temporal and traffic evidence.
- **Explainable decisions:** Supports verification, confidence-based decisions, and inconclusive outcomes.

## Results

| Dataset | Accuracy | Macro-F1 |
|---|---:|---:|
| VNAT | 99.53% | 99.52% |
| ISCXVPN2016 | 98.80% | 98.77% |

The results are reported in the accompanying research paper.

## Datasets

The framework is evaluated on two publicly documented network traffic datasets:

- **VNAT**
- **ISCXVPN2016**

Please refer to the respective dataset providers for access and usage conditions.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Deeplearninglabiiita/Trace-VPN-2.git
cd Trace-VPN-2
```

Install the dependencies listed in the repository:

```bash
pip install -r requirements.txt
```

Refer to the project scripts and configuration files for data preparation, model training, and evaluation instructions.

## Research Paper

**TRACE@VPN: Time-Series Representation for Agentic Classification and Evidence-Driven VPN Traffic Identification**

**Authors:** Agrim Ray, Lokendra Kumar Tiwari, and Vrijendra Singh

## Acknowledgments

The authors acknowledge the Ministry of Electronics and Information Technology (MeitY), the Ministry of Education (MoE), Government of India, and the Indian Institute of Information Technology Allahabad (IIITA) for their support.

## Repository

[https://github.com/Deeplearninglabiiita/Trace-VPN-2](https://github.com/Deeplearninglabiiita/Trace-VPN-2)

## Responsible Use

This project is intended for research and authorized network security analysis. Actual performance may vary across datasets and real-world network environments.
