#!/usr/bin/env python3
"""Build a reproducible discovery-only paper pool from OpenAlex metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


FIELDS = (
    "paper_id",
    "title",
    "year",
    "venue",
    "publication_type",
    "publication_status",
    "layers",
    "source_queries",
    "best_search_rank",
    "authors",
    "doi",
    "primary_url",
    "cited_by_count",
    "open_access",
    "code_url",
    "official_code",
    "stars_query_date",
    "stars",
    "license",
    "simulator",
    "datasets",
    "protocol",
    "baselines",
    "metrics",
    "seeds_variance",
    "prediction_target",
    "planning",
    "uncertainty",
    "migration_target",
    "evidence_status",
    "triage_status",
    "notes",
)


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def request_json(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "cityflow-world-model-research/1.0 (metadata discovery)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def work_key(work: Dict[str, Any]) -> str:
    doi = (work.get("doi") or "").lower().strip()
    return doi or normalize_title(work.get("title") or "")


def compact_work(work: Dict[str, Any]) -> Dict[str, str]:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    authors = []
    for authorship in work.get("authorships") or []:
        name = (authorship.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    identifier = str(work.get("id") or "").rsplit("/", 1)[-1]
    return {
        "paper_id": identifier,
        "title": work.get("title") or "",
        "year": str(work.get("publication_year") or ""),
        "venue": source.get("display_name") or "not_extracted",
        "publication_type": work.get("type") or "not_extracted",
        "publication_status": "candidate_unverified",
        "authors": "; ".join(authors[:8]),
        "doi": work.get("doi") or "",
        "primary_url": location.get("landing_page_url") or work.get("doi") or work.get("id") or "",
        "cited_by_count": str(work.get("cited_by_count") or 0),
        "open_access": str((work.get("open_access") or {}).get("oa_status") or "unknown"),
    }


def empty_extraction_fields() -> Dict[str, str]:
    return {
        "code_url": "not_extracted",
        "official_code": "not_extracted",
        "stars_query_date": "not_extracted",
        "stars": "not_extracted",
        "license": "not_extracted",
        "simulator": "not_extracted",
        "datasets": "not_extracted",
        "protocol": "not_extracted",
        "baselines": "not_extracted",
        "metrics": "not_extracted",
        "seeds_variance": "not_extracted",
        "prediction_target": "not_extracted",
        "planning": "not_extracted",
        "uncertainty": "not_extracted",
        "migration_target": "not_extracted",
        "evidence_status": "indirect_discovery_metadata",
        "triage_status": "pending",
        "notes": "OpenAlex discovery hit; requires primary-source verification",
    }


def read_queries(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def search_query(query: str, per_page: int) -> List[Dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "search": query,
            "filter": "from_publication_date:2018-01-01,to_publication_date:2026-08-13",
            "per-page": per_page,
            "select": (
                "id,doi,title,publication_year,type,primary_location,authorships,"
                "cited_by_count,open_access"
            ),
        }
    )
    return request_json(f"https://api.openalex.org/works?{params}")["results"]


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-query", type=int, default=25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_queries(args.query_matrix)

    raw_rows: List[Dict[str, str]] = []
    merged: Dict[str, Dict[str, str]] = {}
    merged_queries: Dict[str, set] = defaultdict(set)
    merged_layers: Dict[str, set] = defaultdict(set)
    best_rank: Dict[str, int] = {}
    log_rows: List[Tuple[str, str, int]] = []

    for query in queries:
        works = search_query(query["query"], args.per_query)
        log_rows.append((query["query_id"], query["query"], len(works)))
        for rank, work in enumerate(works, start=1):
            key = work_key(work)
            if not key:
                continue
            row = compact_work(work)
            row.update(empty_extraction_fields())
            row["layers"] = query["layer"]
            row["source_queries"] = query["query_id"]
            row["best_search_rank"] = str(rank)
            raw_rows.append(row)
            if key not in merged:
                merged[key] = dict(row)
                best_rank[key] = rank
            best_rank[key] = min(best_rank[key], rank)
            merged_queries[key].add(query["query_id"])
            merged_layers[key].add(query["layer"])
        time.sleep(0.1)

    unique_rows = []
    for key, row in merged.items():
        row["source_queries"] = ";".join(sorted(merged_queries[key]))
        row["layers"] = ";".join(sorted(merged_layers[key]))
        row["best_search_rank"] = str(best_rank[key])
        unique_rows.append(row)
    unique_rows.sort(
        key=lambda row: (
            int(row["best_search_rank"]),
            -int(row["cited_by_count"]),
            row["title"].lower(),
        )
    )

    write_csv(args.output_dir / "search_hits_raw.csv", raw_rows)
    write_csv(args.output_dir / "deduped_papers.csv", unique_rows)
    timestamp = datetime.now(timezone.utc).isoformat()
    with (args.output_dir / "search_log.md").open("w", encoding="utf-8") as handle:
        handle.write("# Search log\n\n")
        handle.write(f"- run_utc: {timestamp}\n")
        handle.write("- discovery_source: OpenAlex Works API\n")
        handle.write("- evidence_level: indirect discovery metadata\n")
        handle.write(f"- raw_hits: {len(raw_rows)}\n")
        handle.write(f"- unique_candidates: {len(unique_rows)}\n")
        handle.write("- primary-source verification: required before claims\n\n")
        handle.write("| query_id | returned | query |\n|---|---:|---|\n")
        for query_id, query_text, count in log_rows:
            handle.write(f"| {query_id} | {count} | {query_text} |\n")
    print(
        json.dumps(
            {
                "raw_hits": len(raw_rows),
                "unique_candidates": len(unique_rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
