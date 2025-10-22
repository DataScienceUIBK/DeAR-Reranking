#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

from pyserini.index import IndexReader
from pyserini.search import get_topics


def main():
    parser = argparse.ArgumentParser(
        description="Convert TREC run files to Listwise-compatible JSON."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Directory containing the TREC run files and where JSON will be written.",
    )
    # Optional: separate output dir (defaults to --path)
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for JSON files (defaults to --path).",
    )
    args = parser.parse_args()

    in_dir = Path(args.path).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trec_paths = [
        "ranked_dl19.trec", "ranked_dl20.trec", "ranked_covid.trec",
        "ranked_dbpedia.trec", "ranked_news.trec", "ranked_nfc.trec",
        "ranked_robust04.trec", "ranked_scifact.trec", "ranked_signal.trec",
        "ranked_touche.trec"
    ]
    output_json_path = [
        "ranked_dl19.json","ranked_dl20.json", "ranked_covid.json",
        "ranked_dbpedia.json", "ranked_news.json", "ranked_nfc.json",
        "ranked_robust04.json", "ranked_scifact.json", "ranked_signal.json",
        "ranked_touche.json"
    ]

    index_name = [
        "msmarco-v1-passage", "msmarco-v1-passage", "beir-v1.0.0-trec-covid.flat",
        "beir-v1.0.0-dbpedia-entity.flat", "beir-v1.0.0-trec-news.flat",
        "beir-v1.0.0-nfcorpus.flat", "beir-v1.0.0-robust04.flat",
        "beir-v1.0.0-scifact.flat", "beir-v1.0.0-signal1m.flat",
        "beir-v1.0.0-webis-touche2020.flat",
    ]
    topic_name = [
        "dl19-passage", "dl20", "beir-v1.0.0-trec-covid-test",
        "beir-v1.0.0-dbpedia-entity-test", "beir-v1.0.0-trec-news-test",
        "beir-v1.0.0-nfcorpus-test", "beir-v1.0.0-robust04-test",
        "beir-v1.0.0-scifact-test", "beir-v1.0.0-signal1m-test",
        "beir-v1.0.0-webis-touche2020-test",
    ]

    for i, trec_name in enumerate(trec_paths):
        trec_path = in_dir / trec_name
        out_path = out_dir / output_json_path[i]

        if not trec_path.exists():
            print(f"⚠️  Skipping: {trec_path} not found.")
            continue

        try:
            print(f"\n=== [{trec_name}] ===")
            print("Loading index...")
            index = IndexReader.from_prebuilt_index(index_name[i])

            print("Loading topics...")
            topics = get_topics(topic_name[i])

            print("Parsing TREC file...")
            results = defaultdict(list)
            with trec_path.open("r", encoding="utf-8") as f:
                for line in f:
                    qid, _, docid, rank, score, _ = line.strip().split()
                    results[qid].append((int(rank), docid, float(score)))

            print("Building Listwise-compatible JSON...")
            final_output = []
            for qid, tuples in results.items():
                # qid may be str or numeric; try both safely
                if qid in topics:
                    query_text = topics[qid]["title"]
                else:
                    try:
                        query_text = topics[int(qid)]["title"]
                    except Exception:
                        # Fallback if topics key mismatch
                        query_text = str(qid)

                query_entry = {"query": query_text, "qid": qid, "hits": []}

                for rank, docid, score in sorted(tuples, key=lambda x: x[0]):
                    raw = index.doc(docid).raw()
                    content_json = json.loads(raw)

                    if "title" in content_json:
                        content = f"Title: {content_json['title']} Content: {content_json.get('text', '')}"
                    else:
                        content = content_json.get("contents", "")

                    content = " ".join(content.split())

                    query_entry["hits"].append({
                        "content": content,
                        "qid": qid,
                        "docid": docid,
                        "rank": rank,
                        "score": score
                    })

                final_output.append(query_entry)

            print(f"Saving to {out_path} ...")
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)

            print("✅ Done.")
        except Exception as e:
            print(f"❌ Error processing {trec_name}: {e}")


if __name__ == "__main__":
    main()
