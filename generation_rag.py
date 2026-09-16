"""Post-generation retrieval augmentation for incident-response drafts.

The first model generation is treated as a draft and retrieval query. Text-RAG
retrieves lexical neighbors from a local corpus; KG-RAG parses the query into an
incident and retrieves its security-graph neighborhood. Both modes return a
revision prompt for a second, final model generation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from incident_log_parser import parse_logs
from kg_rag import SecurityKnowledgeGraph
from response_evaluation import lexical_semantic_similarity


NO_RAG = "none"
TEXT_RAG = "text_rag"
KG_RAG = "kg_rag"
DEFAULT_RAG_OUTPUT = Path("generation_rag_output.json")


@dataclass
class TextDocument:
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGAugmentation:
    mode: str
    query: str
    context_text: str
    revision_prompt: str
    retrieved_items: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _flatten_dataset_mapping(payload: Dict[str, Any]) -> List[TextDocument]:
    documents: List[TextDocument] = []
    instructions = payload.get("instructions", [])
    answers = payload.get("answers", [])
    if instructions and isinstance(instructions[0], list):
        instructions = instructions[0]
    if answers and isinstance(answers[0], list):
        answers = answers[0]
    for index, instruction in enumerate(instructions):
        answer = answers[index] if index < len(answers) else ""
        documents.append(TextDocument(str(index), f"{instruction}\n{answer}".strip()))
    return documents


def load_text_corpus(path: Path) -> List[TextDocument]:
    """Load text retrieval documents from JSON, JSONL, or plain text."""
    raw = path.read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        documents = []
        for index, line in enumerate(line for line in raw.splitlines() if line.strip()):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = line
            documents.append(_document_from_item(item, index))
        return documents
    if isinstance(payload, dict) and "instructions" in payload:
        return _flatten_dataset_mapping(payload)
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("documents", [payload])
    else:
        items = [payload]
    return [_document_from_item(item, index) for index, item in enumerate(items)]


def _document_from_item(item: Any, index: int) -> TextDocument:
    if isinstance(item, str):
        return TextDocument(str(index), item)
    if not isinstance(item, dict):
        return TextDocument(str(index), str(item))
    text = item.get("text") or item.get("content") or item.get("instruction") or item.get("answer")
    if text is None:
        text = json.dumps(item, ensure_ascii=False, sort_keys=True)
    document_id = str(item.get("id", item.get("document_id", index)))
    metadata = {key: value for key, value in item.items() if key not in {"id", "document_id", "text", "content"}}
    return TextDocument(document_id, str(text), metadata)


class TextRAGRetriever:
    """Dependency-free lexical retrieval over a local response corpus."""

    def __init__(self, documents: Sequence[TextDocument], top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.documents = list(documents)
        self.top_k = top_k

    def retrieve(self, query: str) -> tuple[str, List[Dict[str, Any]]]:
        ranked = sorted(
            (
                (lexical_semantic_similarity(query, document.text), index, document)
                for index, document in enumerate(self.documents)
            ),
            key=lambda item: (-item[0], item[1]),
        )[: self.top_k]
        items = [
            {
                "document_id": document.document_id,
                "score": score,
                "text": document.text,
                "metadata": document.metadata,
            }
            for score, _, document in ranked
        ]
        context = "\n\n".join(
            f"[Text reference {item['document_id']} score={item['score']:.4f}]\n{item['text']}"
            for item in items
        )
        return context, items


class PostGenerationRAG:
    """Retrieve grounding context from a draft and build a revision prompt."""

    def __init__(
        self,
        mode: str,
        text_retriever: Optional[TextRAGRetriever] = None,
        kg_depth: int = 2,
    ) -> None:
        if mode not in {TEXT_RAG, KG_RAG}:
            raise ValueError(f"Post-generation RAG mode must be {TEXT_RAG!r} or {KG_RAG!r}.")
        if mode == TEXT_RAG and text_retriever is None:
            raise ValueError("text_rag mode requires a TextRAGRetriever.")
        self.mode = mode
        self.text_retriever = text_retriever
        self.kg_depth = kg_depth

    def prepare_revision(self, instruction: str, draft_generation: str) -> RAGAugmentation:
        query = f"{instruction}\n{draft_generation}".strip()
        if self.mode == TEXT_RAG:
            context_text, items = self.text_retriever.retrieve(query)
            metadata = {"top_k": self.text_retriever.top_k}
        else:
            incident = parse_logs([query], incident_id="post-generation-rag")
            context = SecurityKnowledgeGraph().retrieve_context(incident, depth=self.kg_depth)
            context_dict = asdict(context)
            context_text = context.prompt_context
            items = context_dict.get("nodes", [])
            metadata = {
                "kg_depth": self.kg_depth,
                "incident": context_dict.get("incident", {}),
                "edges": context_dict.get("edges", []),
            }
        revision_prompt = (
            f"{instruction.strip()}\n\n"
            "Revise the draft incident-response answer using only supported details from the retrieved context. "
            "Preserve valid actions, correct unsupported commands, and include targets, evidence, and rollback guidance.\n"
            "<draft_generation>\n"
            f"{draft_generation.strip()}\n"
            "</draft_generation>\n"
            f"<{self.mode}_context>\n"
            f"{context_text}\n"
            f"</{self.mode}_context>"
        )
        return RAGAugmentation(self.mode, query, context_text, revision_prompt, items, metadata)


def save_rag_output(path: Path, augmentation: RAGAugmentation) -> None:
    """Persist the complete retrieval result and revision prompt as local JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(augmentation), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text-RAG or KG-RAG context from a generated draft.")
    parser.add_argument("--mode", choices=(TEXT_RAG, KG_RAG), required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--text-corpus", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--kg-depth", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RAG_OUTPUT,
        help="Local JSON file for the generated RAG context and revision prompt.",
    )
    args = parser.parse_args()
    retriever = None
    if args.mode == TEXT_RAG:
        if not args.text_corpus:
            parser.error("--text-corpus is required for text_rag mode.")
        retriever = TextRAGRetriever(load_text_corpus(args.text_corpus), top_k=args.top_k)
    augmentation = PostGenerationRAG(args.mode, retriever, args.kg_depth).prepare_revision(
        args.instruction, args.generation
    )
    save_rag_output(args.output, augmentation)
    print(json.dumps(asdict(augmentation), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
