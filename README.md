# ReconPotBench 🍯🤖

A multi-round benchmark for evaluating LLM-based honeypots in Linux reconnaissance.

## Overview 🎯

Reconnaissance is the critical first phase of the cyber attack lifecycle, forming the basis for all subsequent attack activities ([MITRE ATT&CK TA0043](https://attack.mitre.org/tactics/TA0043/)). Using large language models (LLMs) as interactive deception mechanisms to counter reconnaissance is a promising paradigm in active defense. However, existing evaluations of LLM-based honeypots are not designed for **multi-round** Linux reconnaissance — they focus on per-response realism without measuring whether entities remain consistent across rounds, or whether the interaction continues to support meaningful follow-up probing.

**ReconPotBench** addresses this gap as the first systematic evaluation benchmark supporting both single-round and multi-round assessment for LLM honeypots.

## Dataset 📊

The full dataset comprises **1,032 three-round interaction test cases** across **8 reconnaissance command families**, following a Network → Software → Hardware hierarchy.

**📌 Partial Release:** This repository currently provides **15% of each reconnaissance category's three-round command-output interaction pairs** (in `data/dataset.json`) for reference. 

### Data Format 🧬

Each entry is a JSON object with a `conversations` array containing 3 rounds of human (reconnaissance command) / gpt (system output) interaction pairs:

```json
{
  "conversations": [
    {"from": "human", "value": "<reconnaissance command>"},
    {"from": "gpt",   "value": "<ground-truth system output>"},
    {"from": "human", "value": "<follow-up command>"},
    {"from": "gpt",   "value": "<ground-truth system output>"},
    {"from": "human", "value": "<follow-up command>"},
    {"from": "gpt",   "value": "<ground-truth system output>"}
  ]
}
```

## Citation 📝
EACL 2027 (Findings) · Accepted

@misc{wang2026reconpotbenchevaluatingllmhoneypots,
      title={ReconPotBench: Evaluating LLM Honeypots in Multi-Round Linux Reconnaissance}, 
      author={Qianli Wang and Junchao Wu and Binkai Ou and Lidia S. Chao and Derek F. Wong},
      year={2026},
      eprint={2611.20528},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2611.20528}, 
}

## License ⚖️

The dataset is released for research purposes.
