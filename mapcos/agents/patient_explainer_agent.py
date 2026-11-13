class PatientExplainerAgent:
    """Plain-language, template-based explanation. Never states a diagnosis."""
    def explain(self, n, criteria, safety):
        met = sum(1 for v in criteria.values() if v == "met")
        lines = ["**What this report is:** a computer-assisted screening summary. It cannot diagnose you. Only a doctor can."]
        if n.get("exclusion_flagged"):
            lines.append("**What we noticed:** one of your thyroid or prolactin blood results is outside the usual range. Other conditions can cause symptoms similar to PCOS, so your doctor needs to check this first.")
        elif safety["route"] != "AUTO_RELEASE_WITH_DISCLAIMER":
            lines.append("**What we noticed:** the results are mixed or not fully clear, so we cannot say much from the computer analysis alone. A doctor should look at them.")
        elif met >= 2:
            lines.append("**What we noticed:** the pattern in your results is similar to what is often seen in PCOS (at least two of three typical features).")
        else:
            lines.append("**What we noticed:** the results do not show enough of the typical features of PCOS. This does not rule it out if you have symptoms.")
        names = {"pcom": "Ultrasound ovary appearance", "ovulatory": "Cycle regularity", "androgen": "Signs of higher male-type hormones"}
        label = {"met": "pattern present", "not_met": "pattern not seen", "uncertain": "unclear", "unknown": "not provided"}
        lines.append("**Three things doctors look at:**")
        for k, v in criteria.items():
            lines.append(f"- {names.get(k, k)}: {label.get(v, v)}")
        lines.append("**What to do next:** share this report with your doctor, especially if you have irregular periods, excess hair growth, or difficulty conceiving. Do not change any medication based on this report.")
        return "\n\n".join(lines)
