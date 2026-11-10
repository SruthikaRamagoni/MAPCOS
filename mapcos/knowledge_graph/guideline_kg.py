
"""
Guideline Knowledge Graph
-------------------------
Small in-memory graph (networkx) holding the diagnostic rules MAPCOS applies:
  PCOS <- 3 Rotterdam criteria (need any 2) <- observable evidence
  PCOS <- must exclude: thyroid dysfunction, hyperprolactinemia
The Synthesis and Exclusion agents read their criteria, evidence strengths and
thresholds from here, so the rules live in ONE place. The same interface can
later be backed by Neo4j without touching the agents.
"""

import networkx as nx

from mapcos.config import PCOM_THRESHOLD, PRL_UPPER, ROTTERDAM_CRITERIA_COUNT, TSH_UPPER

SOURCE = "Rotterdam ESHRE/ASRM consensus, Fertil. Steril. 2004"


class GuidelineKG:
    def __init__(self):
        g = nx.MultiDiGraph()
        g.add_node("PCOS", kind="diagnosis", min_criteria=ROTTERDAM_CRITERIA_COUNT, source=SOURCE)

        criteria = {
            "OvulatoryDysfunction": "Oligo- or anovulation (irregular cycles)",
            "Hyperandrogenism": "Clinical hyperandrogenism (hirsutism) or raised androgens",
            "PCOM": f"Polycystic ovarian morphology (>= {PCOM_THRESHOLD} follicles per ovary)",
        }
        for name, desc in criteria.items():
            g.add_node(name, kind="criterion", description=desc)
            g.add_edge(name, "PCOS", relation="CONTRIBUTES_TO")

        # (evidence, criterion, strength). 'weak' evidence is reported but never
        # satisfies a criterion alone (mild acne must not count as proof of PCOS).
        evidence = [
            ("irregular_cycle", "OvulatoryDysfunction", "strong"),
            ("hair_growth", "Hyperandrogenism", "strong"),
            ("pimples", "Hyperandrogenism", "weak"),
            ("hair_loss", "Hyperandrogenism", "weak"),
            ("cnn_positive", "PCOM", "strong"),
            ("follicle_count_high", "PCOM", "strong"),
        ]
        for ev, crit, strength in evidence:
            g.add_node(ev, kind="evidence")
            g.add_edge(ev, crit, relation="SUPPORTS", strength=strength)

        exclusions = [
            ("ThyroidDysfunction", "tsh", "TSH", "mIU/L", TSH_UPPER,
             "Thyroid dysfunction can cause cycle irregularity; check thyroid function."),
            ("Hyperprolactinemia", "prl", "Prolactin", "ng/mL", PRL_UPPER,
             "Raised prolactin can cause cycle irregularity; investigate before confirming PCOS."),
        ]
        for name, key, marker, unit, upper, advice in exclusions:
            g.add_node(name, kind="exclusion", key=key, marker=marker, unit=unit,
                       upper=upper, advice=advice)
            g.add_edge(name, "PCOS", relation="MUST_BE_EXCLUDED")
        self.g = g

    def min_criteria(self) -> int:
        return self.g.nodes["PCOS"]["min_criteria"]

    def criteria(self) -> dict:
        return {n: d["description"] for n, d in self.g.nodes(data=True) if d["kind"] == "criterion"}

    def strength(self, evidence: str, criterion: str) -> str:
        for _, c, d in self.g.out_edges(evidence, data=True):
            if c == criterion:
                return d["strength"]
        return "none"

    def exclusions(self) -> dict:
        return {n: dict(d) for n, d in self.g.nodes(data=True) if d["kind"] == "exclusion"}

    def triples(self):
        return [(u, d["relation"], v) for u, v, d in self.g.edges(data=True)]
