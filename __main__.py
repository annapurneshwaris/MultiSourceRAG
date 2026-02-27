"""CLI entry point for HeteroRAG.

Usage:
    python -m MultiSourceRAG process     # Run chunking pipeline
    python -m MultiSourceRAG index       # Build all indices
    python -m MultiSourceRAG query "How to configure terminal?"
    python -m MultiSourceRAG ablation    # Run ablation experiments
    python -m MultiSourceRAG judge       # Score results with LLM judge
    python -m MultiSourceRAG train       # Offline router training
    python -m MultiSourceRAG serve       # Start API + UI
"""

import argparse
import sys


def main():
    import config as cfg
    cfg.init_paths()
    cfg.set_seeds()

    parser = argparse.ArgumentParser(
        prog="heterorag",
        description="HeteroRAG: Multi-Source Heterogeneous RAG for Technical Support",
    )
    sub = parser.add_subparsers(dest="command")

    # process
    sub.add_parser("process", help="Run chunking pipeline on raw data")

    # index
    sub.add_parser("index", help="Build FAISS + BM25 indices")

    # query
    q_parser = sub.add_parser("query", help="Run a single query")
    q_parser.add_argument("text", type=str, help="Query text")
    q_parser.add_argument("--config", type=str, default="DBW", help="Source config")
    q_parser.add_argument("--router", type=str, default="heuristic", help="Router type")
    q_parser.add_argument("--no-generate", action="store_true", help="Skip LLM generation")

    # ablation
    a_parser = sub.add_parser("ablation", help="Run ablation experiments")
    a_parser.add_argument("--configs", type=str, default=None)
    a_parser.add_argument("--queries", type=int, default=None)
    a_parser.add_argument("--router", type=str, default="heuristic")
    a_parser.add_argument("--router-sweep", action="store_true")
    a_parser.add_argument("--no-generate", action="store_true")

    # judge
    j_parser = sub.add_parser("judge", help="Run LLM judge on ablation results")
    j_parser.add_argument("--judge-model", type=str, default=cfg.JUDGE_MODEL)
    j_parser.add_argument("--max-results", type=int, default=None)

    # train
    t_parser = sub.add_parser("train", help="Offline router training")
    t_parser.add_argument("--epochs", type=int, default=1)

    # serve
    s_parser = sub.add_parser("serve", help="Start API server + Gradio UI")
    s_parser.add_argument("--host", type=str, default="0.0.0.0")
    s_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "process":
        from processing.run_processing import main as run
        run()

    elif args.command == "index":
        from indexing.run_indexing import main as run
        run()

    elif args.command == "query":
        import json
        from retrieval.pipeline import RetrievalPipeline
        pipeline = RetrievalPipeline(router_type=args.router)
        result = pipeline.process_query(
            query=args.text,
            config=args.config,
            generate=not args.no_generate,
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "ablation":
        from evaluation.ablation_runner import run_ablation
        configs = args.configs.split(",") if args.configs else None
        run_ablation(
            configs=configs,
            max_queries=args.queries,
            generate=not args.no_generate,
            router_type=args.router,
            router_sweep=args.router_sweep,
        )

    elif args.command == "judge":
        from evaluation.judge_runner import run_judge
        run_judge(judge_model=args.judge_model, max_results=args.max_results)

    elif args.command == "train":
        from evaluation.offline_training import train_offline
        train_offline(epochs=args.epochs)

    elif args.command == "serve":
        import uvicorn
        uvicorn.run("api.app:app", host=args.host, port=args.port, reload=False)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
