
"""
Exclusion Agent
---------------
Rotterdam diagnosis requires other causes of the same signs to be excluded.
Checks thyroid (TSH) and prolactin (PRL) against the limits stored in the
knowledge graph and adjusts the synthesis result:

  flag raised            -> PCOS cannot be confirmed until the other cause is ruled out
  all assessed and clear -> diagnosis stands
  value(s) missing       -> diagnosis stands but is marked provisional

Not assessable from this data (listed in every output): non-classic congenital
adrenal hyperplasia (17-OH progesterone), Cushing syndrome, androgen-secreting tumours.
"""

import math

from mapcos.knowledge_graph.guideline_kg import GuidelineKG

NOT_ASSESSABLE = ["non-classic congenital adrenal hyperplasia", "Cushing syndrome",
                  "androgen-secreting tumour"]


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class ExclusionAgent:
    def __init__(self, kg: GuidelineKG = None):
        self.rules = (kg or GuidelineKG()).exclusions()

    def run(self, synthesis: dict, exclusion_values: dict = None) -> dict:
        values = exclusion_values or {}
        flags, clear, not_assessed = [], [], []
        for name, rule in self.rules.items():
            v = values.get(rule["key"])
            if _missing(v):
                not_assessed.append(name)
                continue
            v = float(v)
            if v > rule["upper"]:
                flags.append({"condition": name, "marker": rule["marker"], "value": v,
                              "upper_limit": rule["upper"], "unit": rule["unit"],
                              "advice": rule["advice"]})
            else:
                clear.append(name)

        base = synthesis["diagnosis"]
        if flags:
            status = "other_cause_possible"
            names = ", ".join(f["condition"] for f in flags)
            final = f"{base} - NOT confirmed: rule out {names} first" if base == "PCOS likely" else base
        elif not_assessed:
            status = "partially_assessed" if clear else "not_assessed"
            final = f"{base} (provisional: exclusion tests incomplete)" if base == "PCOS likely" else base
        else:
            status = "clear"
            final = base

        return {
            "agent": "exclusion",
            "status": status,
            "flags": flags,
            "clear": clear,
            "not_assessed": not_assessed,
            "not_assessable_from_data": NOT_ASSESSABLE,
            "final_diagnosis": final,
        }
