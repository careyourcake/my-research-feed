# Small RL Paper Radar

This repository generates a Zotero-compatible RSS 2.0 feed for reinforcement learning and robotics research.

## Public feeds

- Aggregated GitHub Pages feed: `https://careyourcake.github.io/my-research-feed/my_research_feed.xml`
- Raw GitHub backup: `https://raw.githubusercontent.com/careyourcake/my-research-feed/main/my_research_feed.xml`
- Direct arXiv fallback: generated dynamically by the Coze plugin from the user's research keywords.

## Data sources

- Hugging Face Daily Papers
- Papers with Code
- arXiv API

Each source is fetched independently. An outage at one source does not prevent the other sources from updating the feed. Articles are filtered by research keywords and deduplicated by arXiv ID, normalized title, or URL.

The workflow runs every day at 08:00 Asia/Shanghai and can also be started manually from the Actions page.
