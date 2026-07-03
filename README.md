# Incident Response Planning Using a Lightweight Large Language Model with Reduced Hallucination

This repository contains the artifacts related to the paper *"Incident Response Planning Using a Lightweight Large Language Model with Reduced Hallucination"*, which is accepted to The Network and Distributed System Security (NDSS) Symposium 2026. 
We introduce a novel method that enables the effective use of a large language model (LLM) to provide decision support for incident response planning. Our method uses the LLM for translating system logs into effective response plans while addressing its limitations through fine-tuning, information retrieval, and decision-theoretic planning. Unlike prior work, which relies on prompt engineering of frontier models, our method is lightweight and can run on commodity hardware.

<p align="center">
<img src="img/system.png" width="100%" height="100%">
</p>

## NDSS 2026 Paper and presentation.

Paper: [NDSS 2026 proceedings](https://www.ndss-symposium.org/ndss-paper/incident-response-planning-using-a-lightweight-large-language-model-with-reduced-hallucination/).
Video: [NDSS 2026 presentation](https://www.youtube.com/watch?v=TGuNgPEFnwk).

## Artifacts 

- The first public fine-tuning dataset of incidents and response actions. This is the dataset we use to produce the results in the paper. The dataset can be downloaded [here](https://huggingface.co/datasets/kimhammar/CSLE-IncidentResponse-V1).
- The weights of the fine-tuned model, which can be downloaded [here](https://huggingface.co/kimhammar/LLMIncidentResponse).
- Python code for downloading the training dataset (`load_training_dataset.py`).
- Python code for downloading the fine-tuned model and using it to generate an incident response plan (`load_fine_tuned_llm.py`).
- Python code for generating an incident response plan (`response_generation.py`).
- Python code for fine-tuning a new model based on our dataset (`fine_tune_llm.py`).
- [Video demonstration](https://www.youtube.com/watch?v=SCxq2ye-R4Y&) of our LLM-based decision-support system for incident response.


> **Remark 1**: If the artifact evaluator has access to a GPU, they can run <code>response_generation.py</code> and <code>fine_tune_llm.py</code> to test the LLM. If the evaluator **does not** have access to a GPU. We ask the evaluator to check the youtube video linked above to verify the funcionality of the last two Python scripts.

> **Remark 2**:  The software libraries that support our artifacts are open-source and available at https://github.com/Limmen/csle and https://github.com/Limmen/llm_recovery </li>
**(We do not ask the artifact evaluator to verify these libraries.)**


## Requirements

- Python 3.8+
- `load_training_dataset.py` requires 1 GB of storage and a commodity CPU.
- `load_fine_tuned_llm.py`: requires 15 GB of storage and a commodity CPU.
- `response_generation.py`: requires a commodity GPU, e.g., an RTX 8000.
- `fine_tune_llm.py`: requires a commodity GPU, e.g., an RTX 8000.

We have tested the Python scripts on the following platforms: 

- MacOs Sequoia with Python 3.9, 3.10, 3.11, 3.12, and 3.13.
- Ubuntu 22.04 with Python 3.9, 3.10, 3.11, 3.12, and 3.13.

## Installation

To download this repository run the command:
```bash
git clone https://github.com/Limmen/llm_incident_response_ndss26
```
To install the required python libraries, run the command 
```bash
pip install llm_recovery==0.0.13
```
**Note** if you have Python 3.9 or older, run the following command instead:
```bash
pip install llm_recovery==0.0.7
```

## Execution 

### Loading the fine-tuned LLM

Command:
```bash
python load_fine_tuned_llm.py 
```

Expected output:
```bash
⋊> kim@gpu1 ⋊> ~/llm_incident_response_ndss26 on main ◦ python load_fine_tuned_llm.py                   (base) 19:25:01
Loading the fine-tuned incident response LLM.
adapter_config.json: 100%|████████████████████████████████████████████████████████████| 797/797 [00:00<00:00, 4.08MB/s]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 4/4 [00:59<00:00, 14.78s/it]
/home/kim/anaconda3/lib/python3.11/site-packages/torch/cuda/__init__.py:734: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
adapter_model.safetensors:   0%|                                                            | 0.00/201M [00:00<?, ?B/s]/home/kim/anaconda3/lib/python3.11/site-packages/torch/cuda/__init__.py:734: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
adapter_model.safetensors: 100%|████████████████████████████████████████████████████| 201M/201M [00:04<00:00, 47.9MB/s]
tokenizer_config.json: 4.49kB [00:00, 14.2MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████| 11.4M/11.4M [00:01<00:00, 6.47MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████| 371/371 [00:00<00:00, 2.24MB/s]
chat_template.jinja: 2.25kB [00:00, 6.30MB/s]
LLM loaded successfully.
```
> **📝 NOTE:** Depending on your internet connection, the above command may take a couple of minutes to complete. You will see the download progress in your terminal.

### Loading the fine-tuning dataset

Command:
```bash
python load_training_dataset.py
```

Expected output:
```bash
⋊> kim@gpu1 ⋊> ~/llm_incident_response_ndss26 on main ◦ python load_training_dataset.py                 (base) 19:26:24
Loading training dataset.
README.md: 100%|█████████████████████████████████████████████████████████████████████| 33.0/33.0 [00:00<00:00, 187kB/s]
examples_16_june.json: 100%|█████████████████████████████████████████████████████████| 536M/536M [00:03<00:00, 145MB/s]
Generating train split: 1 examples [00:06,  6.32s/ examples]
Training dataset loaded successfully.
```

> **📝 NOTE:** Depending on your internet connection, the above command may take a couple of minutes to complete. You will see the download progress in your terminal.

### Response generation

Command:
```bash
python response_generation.py
```

Expected output (example):
```bash
⋊> kim@gpu1 ⋊> ~/llm_incident_response_ndss26 on main ◦ python response_generation.py                   (base) 19:28:50
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 4/4 [00:05<00:00,  1.47s/it]
/home/kim/anaconda3/lib/python3.11/site-packages/torch/cuda/__init__.py:734: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
/home/kim/anaconda3/lib/python3.11/site-packages/torch/cuda/__init__.py:734: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
I recognize that while the attack is contained, I do not yet have enough information to fully understand or eradicate it. Therefore, I choose to acquire full disk and memory images along with relevant logs, preserving evidence in a forensically sound manner to support analysis.</think>
{
    "Action": "Acquire full disk and memory images of 10.20.11.42 and export DNS, firewall, and NetFlow logs to write-protected storage.",
    "Explanation": "Capturing images and logs secures evidence for later analysis and legal requirements."
}⏎
```


### Action schema event parser

This repository also includes `action_schema_parser.py`, a lightweight parser for normalizing incident-response actions from the [`kimhammar/CSLE-IncidentResponse-V1`](https://huggingface.co/datasets/kimhammar/CSLE-IncidentResponse-V1) dataset or from compatible LLM outputs. The parser converts JSON-like answers such as `Action`/`Explanation` records and free-form response text into a unified action schema with the following modules:

- `action_type`: inferred or explicit response category, such as `containment`, `evidence_acquisition`, `eradication`, `recovery`, `investigation`, `notification`, or `monitoring`.
- `target`: affected hosts, IPs, users, files, services, or other assets found in the action.
- `command`: shell, PowerShell, cloud, Kubernetes, or other operational commands found in the action.
- `precondition`: prerequisites or conditions that should be satisfied before execution.
- `risk`: operational, forensic, or business risks associated with the action.
- `rollback`: backout or recovery instructions.
- `evidence`: logs, images, snapshots, alerts, or other artifacts supporting the action.
- `source_action` and `source_explanation`: original dataset/model text retained for traceability.

Examples:

```bash
python action_schema_parser.py --text '{"Action":"Acquire full disk and memory images of 10.20.11.42 and export DNS, firewall, and NetFlow logs to write-protected storage.","Explanation":"Capturing images and logs secures evidence for later analysis and legal requirements."}'
```

```bash
python action_schema_parser.py --from-dataset --data-file examples_16_june.json --limit 10
```

If you use `--from-dataset`, install the optional Hugging Face dependency first, for example with the repository's existing `llm_recovery` installation path or by installing `datasets` directly.


### Log/alert parsing into structured incident JSON

`incident_log_parser.py` implements the second pipeline stage: it parses raw logs or alerts and extracts indicators of compromise (IOCs), CVEs, assets, services, and likely attack stages into a structured incident JSON document. The parser is deterministic and dependency-light so it can be used before model prompting, response-plan generation, or retrieval.

The generated incident JSON contains:

- `incident_id`, `observed_at`, `summary`, `severity`, and `raw_events` for incident tracking.
- `iocs` with typed values such as IPv4, IPv6, URL, domain, email, hashes, and file paths.
- `cves` normalized as `CVE-YYYY-NNNN...` identifiers.
- `assets` extracted from host, asset, server, endpoint, account, and user fields.
- `services` inferred from service names, ports, and product hints such as SSH, RDP, HTTP(S), DNS, SMB, database, email, Kubernetes, and AWS.
- `attack_stages` mapped to incident-response-friendly stages aligned with MITRE ATT&CK tactics such as initial access, execution, credential access, lateral movement, command and control, exfiltration, and impact.

Example:

```bash
python incident_log_parser.py --incident-id inc-demo --text 'critical ssh brute force login failed src=203.0.113.10 host=web-01 service=sshd CVE-2024-12345'
```

### KG-RAG security context

`kg_rag.py` implements the third pipeline stage: it builds an in-memory security knowledge graph and retrieves a structured KG-RAG context for response planning. The graph connects incident facts with the following node types:

- `incident`
- `ioc`
- `cve`
- `asset`
- `service`
- `attack_stage`
- `mitre_attack`
- `mitigation`
- `security_rule`

The reference graph uses edges such as `mentions_cve`, `affects_asset`, `involves_service`, `has_ioc`, `has_attack_stage`, `maps_to_attack`, `has_mitigation`, and `monitored_by`. The output includes raw graph nodes/edges, ranked mitigation or detection recommendations, and a `prompt_context` string that can be passed to the response-generation LLM.

Examples:

```bash
python kg_rag.py --incident-id inc-kg --text 'CVE-2023-34362 exploit against host=moveit-01 http service followed by data exfil upload'
```

```bash
python incident_log_parser.py --input alerts.log --incident-id inc-001 > incident.json
python kg_rag.py --incident-json incident.json --depth 2
```

#### Code references

- `incident_log_parser.py`: log and alert parsing, IOC/CVE/asset/service/stage extraction, and incident JSON serialization.
- `kg_rag.py`: in-memory knowledge graph construction, MITRE ATT&CK/stage linking, mitigation/security-rule linking, and prompt-context retrieval.
- `action_schema_parser.py`: downstream normalization of generated response actions into the action schema.

#### References for extending the parser and KG-RAG graph

- Hugging Face dataset: [`kimhammar/CSLE-IncidentResponse-V1`](https://huggingface.co/datasets/kimhammar/CSLE-IncidentResponse-V1).
- MITRE ATT&CK Enterprise Matrix: <https://attack.mitre.org/matrices/enterprise/>.
- MITRE ATT&CK Techniques: <https://attack.mitre.org/techniques/enterprise/>.
- MITRE D3FEND countermeasure knowledge graph: <https://d3fend.mitre.org/>.
- NIST National Vulnerability Database (NVD): <https://nvd.nist.gov/>.
- Common Vulnerabilities and Exposures (CVE): <https://www.cve.org/>.
- Sigma generic SIEM rule format: <https://sigmahq.io/>.
- STIX 2.1 specification for cyber threat intelligence objects and relationships: <https://docs.oasis-open.org/cti/stix/v2.1/stix-v2.1.html>.


### Fine-tuning DeepSeek-R1-Distill-Qwen-14B on our incident response dataset

Command:
```bash
python fine_tune_llm.py
```

Expected output:
```bash
⋊> kim@gpu1 ⋊> ~/llm_incident_response_ndss26 on main ⨯ python fine_tune_llm.py                         (base) 20:14:06
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 4/4 [00:35<00:00,  8.85s/it]
Trainable parameters: 50331648
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
Step: 1, Epoch: 0.5000, Progress: 50.0%, Avg_loss=0.5480, LR=0.00095000, Grad_norm=0.3354, minutes: 0.7959
Step: 2, Epoch: 1.0000, Progress: 100.0%, Avg_loss=0.7036, LR=0.00047500, Grad_norm=0.4813, minutes: 1.2205
⋊> kim@gpu2 ⋊> ~/llm_incident_response_ndss26 on main ⨯  
```

## DOI

https://doi.org/10.5281/zenodo.17459636

## Authors

Kim Hammar, Tansu Alpcan, and Emil Lupu. 

Contact: kimham@kth.se

## 🔖 Copyright and license

<p>
<a href="./LICENSE.md">Creative Commons (C) 2025, Kim Hammar, Tansu Alpcan, and Emil Lupu</a>
</p>

<p align="center">

</p>

<p align="center">


</p>

---
<p align="center" style="align-items:center; display:inline-block">
Made with &#10084; &nbsp;
at &nbsp; <a href="https://www.unimelb.edu.au/" target="_blank">
<img align="absmiddle" src="img/unimelb.png" width="25%" height="25%">
</a>
&nbsp; and 
&nbsp;<a href="https://www.imperial.ac.uk/" target="_blank">
<img align="absmiddle" src="img/imperial.png" width="40%" height="40%">
</a>
</p>
