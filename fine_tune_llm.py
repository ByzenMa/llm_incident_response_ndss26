import argparse
from pathlib import Path

from transformers import set_seed

from enriched_training_dataset import (
    DEFAULT_DATA_FILE,
    DEFAULT_KG_RAG_DATA_FILE,
    KG_RAG_MODE,
    ORIGINAL_MODE,
    load_training_examples,
    preprocess_kg_rag_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the incident-response LLM.")
    parser.add_argument(
        "--dataset-mode",
        choices=(ORIGINAL_MODE, KG_RAG_MODE),
        default=ORIGINAL_MODE,
        help="Use the original examples_16_june.json pairs or a local KG-RAG-enriched JSON file.",
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="Original CSLE-IncidentResponse data file; defaults to examples_16_june.json.")
    parser.add_argument("--processed-data-file", default=DEFAULT_KG_RAG_DATA_FILE, help="Local preprocessed KG-RAG JSON file read when --dataset-mode kg_rag is used.")
    parser.add_argument("--preprocess-kg-rag-data", action="store_true", help="Preprocess --data-file into --processed-data-file before loading it for training.")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to fine-tune on; preserves the previous default of 5.")
    parser.add_argument("--kg-depth", type=int, default=2, help="KG neighborhood depth used only during preprocessing.")
    return parser.parse_args()


if __name__ == '__main__':
    # Heavy dependencies stay inside the executable path so helper functions can be unit-tested without a GPU stack.
    from llm_recovery.fine_tuning.examples_dataset import ExamplesDataset
    from llm_recovery.fine_tuning.lora import LORA
    from llm_recovery.load_llm.load_llm import LoadLLM
    import llm_recovery.constants.constants as constants

    args = parse_args()
    seed = 99125
    set_seed(seed)
    device_map = "auto"
    if args.preprocess_kg_rag_data:
        summary = preprocess_kg_rag_dataset(
            data_file=args.data_file,
            output_path=Path(args.processed_data_file),
            limit=args.limit,
            kg_depth=args.kg_depth,
        )
        print(f"Preprocessed KG-RAG training data: {summary}")

    tokenizer, llm = LoadLLM.load_llm(llm_name=constants.LLM.DEEPSEEK_14B_QWEN, device_map=device_map)
    instructions, answers, enrichment_metadata = load_training_examples(
        mode=args.dataset_mode,
        data_file=args.data_file,
        processed_data_file=args.processed_data_file,
        limit=args.limit,
    )
    if enrichment_metadata:
        failed = [item["validation"] for item in enrichment_metadata if not item["validation"]["ok"]]
        if failed:
            raise ValueError(f"KG-RAG enrichment validation failed: {failed}")
        print(f"Using KG-RAG-enriched training dataset with {len(instructions)} examples from {args.processed_data_file}.")
    else:
        print(f"Using original training dataset with {len(instructions)} examples from {args.data_file}.")
    lora_rank = 64
    lora_alpha = 128
    lora_dropout = 0.05
    llm = LORA.setup_llm_for_fine_tuning(llm=llm, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    dataset = ExamplesDataset(instructions=instructions, answers=answers, tokenizer=tokenizer)
    lr = 0.00095
    per_device_batch_size = 1
    num_train_epochs = 1
    prompt_logging_frequency = 50
    max_generation_tokens = 6000
    logging_steps = 1
    running_average_window = 100
    temperature = 0.6
    save_steps = 25
    save_limit = 3
    gradient_accumulation_steps = 1
    progress_save_frequency = 10
    LORA.supervised_fine_tuning(llm=llm, dataset=dataset, learning_rate=lr,
                                per_device_train_batch_size=per_device_batch_size,
                                num_train_epochs=num_train_epochs, logging_steps=logging_steps, prompts=instructions,
                                answers=answers,
                                prompt_logging=True,
                                running_average_window=running_average_window,
                                max_generation_tokens=max_generation_tokens,
                                prompt_logging_frequency=prompt_logging_frequency, temperature=temperature,
                                save_steps=save_steps, save_limit=save_limit,
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                progress_save_frequency=progress_save_frequency, seed=seed)
