from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mapcos.knowledge_graph.guideline_corpus import CORPUS

class GuidelineRAGAgent:
    """TF-IDF retrieval over guideline passages. Retrieval only, no text generation."""
    def __init__(self, corpus=None):
        self.corpus = corpus or CORPUS
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.mat = self.vec.fit_transform([p["text"] for p in self.corpus])

    def build_query(self, n, safety):
        q, topics = ["PCOS Rotterdam criteria"], {"rotterdam"}
        codes = {r["code"] for r in safety["reasons"]}
        if n.get("follicle_count") is not None or {"FOLLICLE_BORDERLINE", "CNN_FOLLICLE_DISAGREE"} & codes:
            q.append("polycystic ovarian morphology follicle number threshold ultrasound"); topics.add("pcom")
        if "EXCLUSION_FLAG" in codes or "EXCLUSION_MISSING" in codes:
            q.append("exclude thyroid prolactin other causes"); topics.add("exclusion")
        if "LAB_MISSING" in codes or "LAB_IMAGING_CONFLICT" in codes:
            q.append("AMH LH FSH supportive"); topics.update({"labs", "pcom"})
        if "SYMPTOMS_MISSING" in codes:
            q.append("irregular cycles hirsutism hyperandrogenism"); topics.update({"ovulation", "androgen"})
        if safety["route"] != "AUTO_RELEASE_WITH_DISCLAIMER":
            topics.add("safety")
        return " ".join(q), topics

    def retrieve(self, n, safety, k=5):
        query, topics = self.build_query(n, safety)
        sims = cosine_similarity(self.vec.transform([query]), self.mat)[0]
        scored = [(s + (0.25 if p["topic"] in topics else 0.0), p) for s, p in zip(sims, self.corpus)]
        scored.sort(key=lambda x: -x[0])
        return [{**p, "score": round(float(s), 3)} for s, p in scored[:k]]
