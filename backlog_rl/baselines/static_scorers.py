"""
backlog_rl.baselines.static_scorers
------------------------------------
RICE and WSJF baseline implementations, exactly as specified in
MR603 Chapter 3, Section 3.4.

Both are pure, stateless scoring functions -- no memory is carried
between planning cycles, which is the deliberate methodological
point: these are the static frameworks this thesis contrasts against
a learning RL policy.
"""

from enum import IntEnum


class Tier(IntEnum):
    LATER = 0
    NEXT = 1
    NOW = 2


def rice_score(item, confidence_risk_threshold=0.7):
    """
    RICE = (Reach x Impact x Confidence) / Effort   (Chapter 3, Section 3.4.1)

    Reach     <- demand_signal
    Impact    <- business_value
    Confidence<- 0.8 by default, reduced to 0.5 if risk > threshold
    Effort    <- effort (normalised); a small epsilon avoids div-by-zero
    """
    reach = item.demand_signal
    impact = item.business_value
    confidence = 0.5 if item.risk > confidence_risk_threshold else 0.8
    effort = max(item.effort, 1e-6)
    return (reach * impact * confidence) / effort


def wsjf_score(item):
    """
    WSJF = Cost of Delay / Job Duration   (Chapter 3, Section 3.4.2)

    Cost of Delay <- business_value + urgency + (1 - risk)
    Job Duration  <- effort (normalised); a small epsilon avoids div-by-zero
    """
    cost_of_delay = item.business_value + item.urgency + (1.0 - item.risk)
    duration = max(item.effort, 1e-6)
    return cost_of_delay / duration


def assign_tiers(items, score_fn, capacity_points, next_band_fraction=0.5):
    """
    Rank items by descending score, then greedily assign 'Now' up to
    available capacity (in raw story points), the next band of items
    (up to next_band_fraction x capacity worth of additional effort)
    to 'Next', and everything else to 'Later'.

    Returns: dict item_id -> Tier, list of (item, score) sorted desc.
    """
    scored = sorted(((it, score_fn(it)) for it in items), key=lambda x: x[1], reverse=True)

    tiers = {}
    used_now = 0
    used_next = 0
    next_budget = capacity_points * next_band_fraction

    for item, score in scored:
        if used_now + item.effort_raw <= capacity_points:
            tiers[item.item_id] = Tier.NOW
            used_now += item.effort_raw
        elif used_next + item.effort_raw <= next_budget:
            tiers[item.item_id] = Tier.NEXT
            used_next += item.effort_raw
        else:
            tiers[item.item_id] = Tier.LATER

    return tiers, scored


if __name__ == "__main__":
    from backlog_rl.env.generator import BacklogGenerator

    gen = BacklogGenerator(seed=42)
    backlog = gen.generate_backlog(10)
    capacity = 20  # story points available this sprint

    rice_tiers, rice_ranked = assign_tiers(backlog, rice_score, capacity)
    wsjf_tiers, wsjf_ranked = assign_tiers(backlog, wsjf_score, capacity)

    print(f"Sprint capacity: {capacity} story points\n")
    print(f"{'Item':30s} {'Effort':>7s} {'RICE':>7s} {'Tier(RICE)':>11s} {'WSJF':>7s} {'Tier(WSJF)':>11s}")
    for item in backlog:
        r_score = next(s for it, s in rice_ranked if it.item_id == item.item_id)
        w_score = next(s for it, s in wsjf_ranked if it.item_id == item.item_id)
        print(f"{item.name:30s} {item.effort_raw:>5d}pt {r_score:7.3f} {rice_tiers[item.item_id].name:>11s} "
              f"{w_score:7.3f} {wsjf_tiers[item.item_id].name:>11s}")
