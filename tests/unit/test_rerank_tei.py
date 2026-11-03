import json

import httpx
import pytest

from rag_engine.rerank.tei import RerankError, TEIReranker


def make(handler, **kw) -> TEIReranker:
    client = httpx.Client(base_url="http://tei", transport=httpx.MockTransport(handler))
    return TEIReranker("http://tei", client=client, **kw)


def test_scores_returned_in_input_order():
    def handler(req):
        body = json.loads(req.content)
        assert body["query"] == "q" and body["truncate"] is True
        # TEI trả theo điểm giảm dần kèm index
        return httpx.Response(200, json=[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])

    assert make(handler).rerank("q", ["a", "b"]) == [0.1, 0.9]


def test_batches_requests_over_limit():
    sizes = []

    def handler(req):
        texts = json.loads(req.content)["texts"]
        sizes.append(len(texts))
        return httpx.Response(200, json=[{"index": i, "score": float(i)} for i in range(len(texts))])

    scores = make(handler, batch_size=32).rerank("q", [str(i) for i in range(40)])
    assert sizes == [32, 8] and len(scores) == 40 and scores[32] == 0.0


def test_http_error_raises():
    with pytest.raises(RerankError, match="413"):
        make(lambda req: httpx.Response(413, json={"error": "batch size"})).rerank("q", ["a"])


def test_missing_scores_raise():
    with pytest.raises(RerankError):
        make(lambda req: httpx.Response(200, json=[{"index": 0, "score": 1.0}])).rerank("q", ["a", "b"])


def test_empty_input_makes_no_request():
    def handler(req):
        raise AssertionError("không được gọi")

    assert make(handler).rerank("q", []) == []
