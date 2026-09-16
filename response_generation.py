import argparse
import json
from dataclasses import asdict
import random
import threading
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from enriched_training_dataset import DEFAULT_DATA_FILE, DEFAULT_KG_RAG_DATA_FILE, KG_RAG_MODE, ORIGINAL_MODE, load_training_examples
from response_post_processor import GenerationPostProcessor, load_context


DEFAULT_GENERATION_MODEL = "kimhammar/LLMIncidentResponse"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and optionally post-process an incident-response plan.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_GENERATION_MODEL, help="HF model id or local fine-tuned model directory.")
    parser.add_argument("--dataset-mode", choices=(ORIGINAL_MODE, KG_RAG_MODE), default=ORIGINAL_MODE, help="Select original or preprocessed KG-RAG examples for prompt sampling.")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="Original examples_16_june.json source.")
    parser.add_argument("--processed-data-file", default=DEFAULT_KG_RAG_DATA_FILE, help="Local KG-RAG preprocessed examples file.")
    parser.add_argument("--instruction", help="Explicit instruction/log prompt. If omitted, one prompt is sampled from the selected dataset.")
    parser.add_argument("--kg-context", type=Path, help="Optional SecurityContext JSON for generation post-processing.")
    parser.add_argument("--enable-post-processing", action="store_true", help="Validate generated actions for CVE authenticity, command syntax, policy constraints, and attack-path consistency.")
    parser.add_argument("--post-processed-output", type=Path, help="Optional local JSON file for the post-processing report.")
    parser.add_argument("--allowed-cve", action="append", default=[], help="Trusted CVE identifier for post-processing; can be repeated.")
    parser.add_argument("--allow-external-cves", action="store_true", help="Do not warn when a syntactically valid CVE is absent from KG-RAG context.")
    parser.add_argument("--max-new-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-sample", dest="do_sample", action="store_false", default=True)
    return parser.parse_args()


def select_instruction(args) -> str:
    if args.instruction:
        return args.instruction
    if args.dataset_mode == KG_RAG_MODE:
        instructions, _, _ = load_training_examples(
            mode=KG_RAG_MODE,
            processed_data_file=args.processed_data_file,
            data_file=args.data_file,
        )
        return random.choice(instructions)
    dataset = load_dataset("kimhammar/CSLE-IncidentResponse-V1", data_files=args.data_file)
    instructions = dataset["train"]["instructions"][0]
    return random.choice(instructions)


def generate_response(model, tokenizer, instruction: str, device: str, max_new_tokens: int, temperature: float, do_sample: bool) -> str:
    inputs = tokenizer(instruction, return_tensors="pt").to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    thread = threading.Thread(target=model.generate, kwargs={**inputs, **gen_kwargs, "streamer": streamer})
    thread.start()
    generated_chunks = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_chunks.append(new_text)
    return "".join(generated_chunks)


def post_process_generation(generation: str, kg_context_path: Path | None = None, allowed_cves=None, allow_external_cves: bool = False):
    processor = GenerationPostProcessor(
        allowed_cves=allowed_cves or [],
        require_context_cve_match=not allow_external_cves,
    )
    return processor.process(generation, kg_context=load_context(kg_context_path))


if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.eval()
    instruction = select_instruction(args)
    generated = generate_response(
        model=model,
        tokenizer=tokenizer,
        instruction=instruction,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )
    if args.enable_post_processing:
        report = post_process_generation(
            generated,
            kg_context_path=args.kg_context,
            allowed_cves=args.allowed_cve,
            allow_external_cves=args.allow_external_cves,
        )
        report_json = json.dumps(asdict(report), indent=2, ensure_ascii=False)
        print("\n\n<PostProcessingReport>")
        print(report_json)
        print("</PostProcessingReport>")
        if args.post_processed_output:
            args.post_processed_output.write_text(report_json, encoding="utf-8")
