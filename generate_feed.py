#!/usr/bin/env python3
"""Build a resilient RSS feed for reinforcement-learning and robotics papers."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_KEYWORDS = [
    "reinforcement learning",
    "sim-to-real",
    "legged locomotion",
    "whole-body control",
    "offline RL",
    "world model",
    "robot foundation model",
    "imitation learning",
    "robot learning",
    "humanoid",
    "locomotion",
]

USER_AGENT = "Small-RL-Paper-Radar/1.0 (public research feed)"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class Paper:
    title: str
    url: str
    summary: str
    published: datetime
    source: str
    authors: str = ""
    arxiv_id: str = ""


def fetch_bytes(url: str, timeout: int = 20) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/atom+xml, application/xml"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_datetime(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def matches_keywords(paper: Paper, keywords: list[str]) -> bool:
    haystack = f"{paper.title} {paper.summary}".lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def extract_arxiv_id(value: str) -> str:
    match = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:)(\d{4}\.\d{4,5})(?:v\d+)?", value, re.I)
    return match.group(1) if match else ""


def fetch_huggingface(keywords: list[str], limit: int = 100) -> list[Paper]:
    payload = json.loads(fetch_bytes("https://huggingface.co/api/daily_papers?limit=100"))
    papers: list[Paper] = []
    for row in payload[:limit]:
        paper_data = row.get("paper", row) if isinstance(row, dict) else {}
        arxiv_id = normalize_text(paper_data.get("id") or paper_data.get("paperId"))
        title = normalize_text(paper_data.get("title"))
        if not title or not arxiv_id:
            continue
        paper = Paper(
            title=title,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            summary=normalize_text(paper_data.get("summary") or paper_data.get("abstract")),
            published=parse_datetime(row.get("publishedAt") or paper_data.get("publishedAt")),
            source="Hugging Face Daily Papers",
            authors=", ".join(
                normalize_text(author.get("name") if isinstance(author, dict) else author)
                for author in paper_data.get("authors", [])
            ),
            arxiv_id=arxiv_id,
        )
        if matches_keywords(paper, keywords):
            papers.append(paper)
    return papers


def fetch_papers_with_code(keywords: list[str], per_keyword: int = 10) -> list[Paper]:
    papers: list[Paper] = []
    # The API is queried by a small subset to keep the daily job fast and polite.
    for keyword in keywords[:6]:
        query = urllib.parse.urlencode({"q": keyword, "items_per_page": per_keyword})
        payload = json.loads(fetch_bytes(f"https://paperswithcode.com/api/v1/papers/?{query}"))
        for row in payload.get("results", []):
            title = normalize_text(row.get("title"))
            url = normalize_text(row.get("url_abs") or row.get("url_pdf") or row.get("paper_url"))
            arxiv_id = normalize_text(row.get("arxiv_id")) or extract_arxiv_id(url)
            if not url and arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id}"
            if not title or not url:
                continue
            paper = Paper(
                title=title,
                url=url,
                summary=normalize_text(row.get("abstract")),
                published=parse_datetime(row.get("published") or row.get("proceeding")),
                source="Papers with Code",
                authors=normalize_text(row.get("authors")),
                arxiv_id=arxiv_id,
            )
            if matches_keywords(paper, keywords):
                papers.append(paper)
    return papers


def build_arxiv_api_url(keywords: list[str], max_results: int = 50) -> str:
    # Phrase queries improve precision while arXiv remains an independent fallback.
    search_query = " OR ".join(f'all:"{keyword}"' for keyword in keywords)
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    return f"https://export.arxiv.org/api/query?{params}"


def fetch_arxiv(keywords: list[str], max_results: int = 50) -> list[Paper]:
    root = ET.fromstring(fetch_bytes(build_arxiv_api_url(keywords, max_results), timeout=30))
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        entry_url = normalize_text(entry.findtext("atom:id", default="", namespaces=ARXIV_NS))
        arxiv_id = extract_arxiv_id(entry_url)
        paper = Paper(
            title=normalize_text(entry.findtext("atom:title", default="", namespaces=ARXIV_NS)),
            url=entry_url,
            summary=normalize_text(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS)),
            published=parse_datetime(entry.findtext("atom:published", default="", namespaces=ARXIV_NS)),
            source="arXiv",
            authors=", ".join(
                normalize_text(author.findtext("atom:name", default="", namespaces=ARXIV_NS))
                for author in entry.findall("atom:author", ARXIV_NS)
            ),
            arxiv_id=arxiv_id,
        )
        if paper.title and paper.url and matches_keywords(paper, keywords):
            papers.append(paper)
    return papers


def deduplicate(papers: Iterable[Paper]) -> list[Paper]:
    selected: dict[str, Paper] = {}
    for paper in sorted(papers, key=lambda item: item.published, reverse=True):
        normalized_title = re.sub(r"[^a-z0-9]+", "", paper.title.lower())
        key = paper.arxiv_id or normalized_title or paper.url.lower()
        if key not in selected:
            selected[key] = paper
    return list(selected.values())


def cdata(value: str) -> str:
    return f"<![CDATA[{value.replace(']]>', ']]]]><![CDATA[>')}]]>"


def render_rss(papers: list[Paper], statuses: dict[str, str], output: Path, max_items: int) -> None:
    now = datetime.now(timezone.utc)
    status_text = "; ".join(f"{name}: {status}" for name, status in statuses.items())
    items: list[str] = []
    for paper in papers[:max_items]:
        description = (
            f"<p><strong>Source:</strong> {html.escape(paper.source)}</p>"
            f"<p><strong>Authors:</strong> {html.escape(paper.authors or 'Not provided')}</p>"
            f"<p>{html.escape(paper.summary)}</p>"
        )
        guid = paper.arxiv_id or hashlib.sha256(paper.url.encode("utf-8")).hexdigest()
        items.append(
            "\n".join(
                [
                    "    <item>",
                    f"      <title>{html.escape(paper.title)}</title>",
                    f"      <link>{html.escape(paper.url)}</link>",
                    f"      <guid isPermaLink=\"false\">{html.escape(guid)}</guid>",
                    f"      <pubDate>{format_datetime(paper.published)}</pubDate>",
                    f"      <category>{html.escape(paper.source)}</category>",
                    f"      <description>{cdata(description)}</description>",
                    "    </item>",
                ]
            )
        )

    xml = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<rss version="2.0">',
            "  <channel>",
            "    <title>Small RL Paper Radar</title>",
            "    <link>https://github.com/careyourcake/my-research-feed</link>",
            "    <description>Daily reinforcement learning and robotics papers. "
            + html.escape(status_text)
            + "</description>",
            "    <language>en</language>",
            f"    <lastBuildDate>{format_datetime(now)}</lastBuildDate>",
            "    <ttl>1440</ttl>",
            *items,
            "  </channel>",
            "</rss>",
            "",
        ]
    )
    output.write_text(xml, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="my_research_feed.xml")
    parser.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS)
    parser.add_argument("--max-items", type=int, default=60)
    args = parser.parse_args()

    papers: list[Paper] = []
    statuses: dict[str, str] = {}
    sources = [
        ("Hugging Face", fetch_huggingface),
        ("Papers with Code", fetch_papers_with_code),
        ("arXiv", fetch_arxiv),
    ]
    for name, fetcher in sources:
        try:
            results = fetcher(args.keywords)
            papers.extend(results)
            statuses[name] = f"ok ({len(results)} items)"
        except (OSError, ValueError, KeyError, ET.ParseError, urllib.error.URLError) as exc:
            statuses[name] = f"unavailable ({type(exc).__name__})"
            print(f"{name}: {exc}", file=sys.stderr)

    unique_papers = deduplicate(papers)
    render_rss(unique_papers, statuses, Path(args.output), args.max_items)
    print(json.dumps({"items": len(unique_papers), "sources": statuses}, ensure_ascii=False))
    # Produce a valid feed even during partial outages; fail only if every source is empty.
    return 0 if unique_papers else 1


if __name__ == "__main__":
    raise SystemExit(main())
