import pytest
from mapcos.knowledge_graph.graph import LocalGraph
from mapcos.rag.retriever import GuidelineRetriever, chunk_text


def test_graph_all_edges_sourced_and_exclusions():
    g = LocalGraph()
    assert all(e["sources"] for e in g.edges)
    names = " ".join(x["condition"] for x in g.exclusions_for())
    assert "Thyroid" in names and "prolactin" in names.lower()
    assert len(g.criteria()) == 3


def test_graph_rejects_unsourced_edge(tmp_path):
    import json, shutil, os
    from mapcos.knowledge_graph import graph
    d = json.load(open(os.path.join(graph.HERE, "seed.json")))
    d["edges"][0]["sources"] = []
    p = tmp_path / "s.json"; p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        LocalGraph(seed_path=str(p))


def test_retrieval_relevant_with_citation():
    r = GuidelineRetriever.default().retrieve("AMH as alternative to ultrasound for polycystic ovarian morphology in adults")
    assert r["status"] == "OK" and r["passages"][0]["citation"]["title"]
    assert "AMH" in r["passages"][0]["text"]


def test_follicle_threshold_query():
    r = GuidelineRetriever.default().retrieve("how many follicles per ovary define PCOM")
    assert r["status"] == "OK" and any("20" in p["text"] for p in r["passages"])


def test_irrelevant_query_insufficient_evidence():
    r = GuidelineRetriever.default().retrieve("best pizza toppings for football season")
    assert r["status"] == "INSUFFICIENT_EVIDENCE" and r["passages"] == []


def test_chunking_and_empty_index():
    assert chunk_text("One. Two. Three. Four. Five.", "X", "t")
    with pytest.raises(ValueError):
        GuidelineRetriever().build()
