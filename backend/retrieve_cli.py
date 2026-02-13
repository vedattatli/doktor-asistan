from __future__ import annotations

import argparse
import sys

from backend.vector_store import (
    build_embedding_index,
    build_index,
    load_chunks_from_jsonl,
    search_embedding,
    search_index,
)


def _normalize_preview(text: str, max_chars: int = 200) -> str:
    return " ".join(text.split())[:max_chars]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query retrieval index from chunks.jsonl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build retrieval index")
    build_parser.add_argument("--chunks", default="out/chunks.jsonl", help="Path to chunks.jsonl")
    build_parser.add_argument("--indexdir", default="out/index", help="Directory where index files are written")
    build_parser.add_argument(
        "--retrieval",
        choices=["tfidf", "embedding"],
        default="tfidf",
        help="Index type to build",
    )

    search_parser = subparsers.add_parser("search", help="Search in retrieval index")
    search_parser.add_argument("--indexdir", default="out/index", help="Directory containing index files")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--topk", type=int, default=5, help="Number of top results")
    search_parser.add_argument(
        "--retrieval",
        choices=["tfidf", "embedding"],
        default="tfidf",
        help="Retrieval type to use",
    )

    return parser


def _cmd_build(args: argparse.Namespace) -> int:
    used_retrieval = args.retrieval
    if args.retrieval == "embedding":
        try:
            chunks = load_chunks_from_jsonl(args.chunks)
            info = build_embedding_index(chunks, args.indexdir)
        except Exception:
            info = build_index(args.chunks, args.indexdir)
            used_retrieval = "tfidf"
    else:
        info = build_index(args.chunks, args.indexdir)

    print("Index build complete.")
    print(f"retrieval: {used_retrieval}")
    print(f"chunks: {info['chunk_count']}")

    if used_retrieval == "embedding":
        print(f"embedding_model: {info['embedding_model']}")
        print(f"embedding_dim: {info['embedding_dim']}")
        print(f"embeddings: {info['embeddings_path']}")
        print(f"embeddings_meta: {info['embeddings_meta_path']}")
        if "skipped_garbage" in info:
            print(f"skipped_garbage: {info['skipped_garbage']}")
        return 0

    print(f"vocab_size: {info['vocab_size']}")
    print(f"matrix_shape: ({info['matrix_rows']}, {info['matrix_cols']})")
    print(f"vectorizer: {info['vectorizer_path']}")
    print(f"matrix: {info['matrix_path']}")
    print(f"meta: {info['meta_path']}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    if args.retrieval == "embedding":
        try:
            results = search_embedding(args.indexdir, args.query, args.topk)
        except Exception:
            results = search_index(args.indexdir, args.query, args.topk)
    else:
        results = search_index(args.indexdir, args.query, args.topk)

    if not results:
        print("No results.")
        return 0

    for rank, item in enumerate(results, start=1):
        preview = _normalize_preview(item["text_masked"], max_chars=200)
        print(
            f"{rank}. score={item['score']:.4f} | doc_name={item['doc_name']} | "
            f"page={item['page']} | chunk_id={item['chunk_id']}"
        )
        print(f"   preview: {preview}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "build":
            return _cmd_build(args)
        if args.command == "search":
            return _cmd_search(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
