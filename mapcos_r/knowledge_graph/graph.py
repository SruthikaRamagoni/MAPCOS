"""Medical guideline knowledge graph.

Default backend = local JSON (no server needed; used for tests/demo).
Optional backend = Neo4j: `Neo4jGraph(uri,user,password).load(LocalGraph())` pushes the same nodes/edges.
Every edge carries source ids that resolve to sources.json (title, venue, date, URL). No edge without a source.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

HERE = os.path.dirname(__file__)


class LocalGraph:
    def __init__(self, seed_path: Optional[str] = None, sources_path: Optional[str] = None):
        with open(seed_path or os.path.join(HERE, "seed.json")) as f:
            d = json.load(f)
        with open(sources_path or os.path.join(HERE, "sources.json")) as f:
            self.sources: Dict[str, dict] = json.load(f)
        self.nodes = {n["id"]: n for n in d["nodes"]}
        self.edges: List[dict] = d["edges"]
        self.validate()

    def validate(self) -> None:
        for e in self.edges:
            if not e.get("sources"):
                raise ValueError(f"edge without source: {e}")
            for s in e["sources"]:
                if s not in self.sources:
                    raise ValueError(f"unknown source {s}")

    def neighbors(self, node_id: str, rel: Optional[str] = None) -> List[dict]:
        return [e for e in self.edges if (e["from"] == node_id or e["to"] == node_id) and (rel is None or e["rel"] == rel)]

    def exclusions_for(self, condition: str = "PCOS") -> List[dict]:
        out = []
        for e in self.edges:
            if e["rel"] == "EXCLUDE_BEFORE_DIAGNOSIS" and e["to"] == condition:
                out.append({"condition": self.nodes[e["from"]]["label"], "sources": [self.sources[s] for s in e["sources"]],
                            "note": e.get("note", "")})
        return out

    def criteria(self) -> List[dict]:
        return [{"id": e["to"], "label": self.nodes[e["to"]]["label"], "sources": e["sources"]}
                for e in self.edges if e["rel"] == "HAS_CRITERION"]

    def citation(self, source_id: str) -> dict:
        return {"id": source_id, **self.sources[source_id]}


class Neo4jGraph:  # pragma: no cover - needs a running server
    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # optional dependency
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def load(self, g: LocalGraph) -> None:
        with self.driver.session() as s:
            for n in g.nodes.values():
                s.run("MERGE (x:Entity {id:$id}) SET x.type=$type, x.label=$label", **n)
            for e in g.edges:
                s.run("MATCH (a:Entity {id:$f}), (b:Entity {id:$t}) "
                      "MERGE (a)-[r:REL {rel:$rel}]->(b) SET r.sources=$src, r.note=$note",
                      f=e["from"], t=e["to"], rel=e["rel"], src=e["sources"], note=e.get("note", ""))
