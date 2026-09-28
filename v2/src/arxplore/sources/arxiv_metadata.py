import asyncio
from collections.abc import Sequence

from arxplore.sources.arxiv import ArxivClient, ArxivPaper, normalize_arxiv_id, parse_arxiv_feed

METADATA_RETRY_COUNT = 4
METADATA_MIN_RETRY_DELAY_SECONDS = 2.0
ID_LIST_SEPARATOR = ","


def unique_arxiv_ids(arxiv_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(normalize_arxiv_id(arxiv_id) for arxiv_id in arxiv_ids if arxiv_id))


def metadata_batches(arxiv_ids: Sequence[str], batch_size: int) -> list[list[str]]:
    unique_ids = unique_arxiv_ids(arxiv_ids)
    size = max(1, batch_size)
    return [unique_ids[index : index + size] for index in range(0, len(unique_ids), size)]


async def fetch_arxiv_metadata(
    client: ArxivClient,
    arxiv_ids: Sequence[str],
    *,
    batch_size: int,
) -> dict[str, ArxivPaper]:
    batches = metadata_batches(arxiv_ids, batch_size)
    results: dict[str, ArxivPaper] = {}
    for position, batch in enumerate(batches):
        response = await client.get_with_retries(
            {"id_list": ID_LIST_SEPARATOR.join(batch)},
            timeout=client.timeout_seconds,
            retry_count=METADATA_RETRY_COUNT,
            retry_delay_seconds=max(METADATA_MIN_RETRY_DELAY_SECONDS, float(client.delay_seconds)),
        )
        results.update(parse_arxiv_feed(response.text))
        if position + 1 < len(batches) and client.delay_seconds > 0:
            await asyncio.sleep(client.delay_seconds)
    return results
