"""
backlog_rl.explain.template_explainer
----------------------------------------
Prototype template-grounded explainability generator, per Chapter 3,
Section 3.6 and the worked example in the Week 5 MDP Specification,
Section 3.3.

Given a backlog item and the tier it was assigned, this inspects the
SAME state features the decision-maker (baseline or, from Week 9, the
trained RL policy) used, and produces a plain-English justification.

No generative language model is used -- this is a rule-based
prototype, deliberately scoped below the MR604 extension objective
of a fully tuned LLM orchestration layer.
"""

from backlog_rl.baselines.static_scorers import Tier

URGENCY_THRESHOLD = 0.7
HIGH_VALUE_THRESHOLD = 0.6
HIGH_RISK_THRESHOLD = 0.6


def explain_decision(item, tier: Tier, capacity_overrun_flag: bool = False) -> str:
    name = item.name
    high_value = item.business_value >= HIGH_VALUE_THRESHOLD
    has_dependency = item.dependency_count > 0
    is_urgent = item.urgency >= URGENCY_THRESHOLD
    is_risky = item.risk >= HIGH_RISK_THRESHOLD

    if tier == Tier.NOW:
        if high_value and has_dependency:
            return (f'"{name}" was prioritised for this sprint for its high business value '
                     f'({item.business_value:.2f}), despite an unresolved dependency — flagged as a risk.')
        if is_urgent:
            return (f'"{name}" was prioritised primarily due to its approaching urgency '
                     f'({item.urgency:.2f}).')
        if high_value and not is_risky:
            return (f'"{name}" was prioritised for its high value ({item.business_value:.2f}) '
                     f'with low delivery risk and no blocking dependencies.')
        if high_value and is_risky:
            return (f'"{name}" was prioritised for its value, though it carries elevated '
                     f'technical risk ({item.risk:.2f}) — worth monitoring closely.')
        return (f'"{name}" was prioritised this sprint based on its overall value-to-effort balance.')

    if tier == Tier.NEXT:
        if has_dependency:
            return (f'"{name}" was deferred to the next cycle — it has {item.dependency_count} '
                     f'unresolved dependenc{"y" if item.dependency_count == 1 else "ies"} that should '
                     f'resolve first.')
        return (f'"{name}" was queued for the next cycle: solid value ({item.business_value:.2f}) '
                f'but not urgent enough to displace this sprint\'s top items.')

    # LATER
    if item.business_value < 0.3:
        return f'"{name}" was deprioritised — low estimated business value ({item.business_value:.2f}).'
    if is_risky and not high_value:
        return (f'"{name}" was deprioritised — moderate value but elevated risk '
                 f'({item.risk:.2f}) made it a weaker candidate this cycle.')
    return (f'"{name}" was deprioritised this cycle in favour of higher-scoring items; '
            f'it remains in the backlog for a future sprint.')


if __name__ == "__main__":
    from backlog_rl.env.generator import BacklogGenerator
    from backlog_rl.baselines.static_scorers import wsjf_score, assign_tiers

    print("=" * 78)
    print("EXPLAINABILITY PROTOTYPE DEMO")
    print("(decisions shown here are WSJF baseline tiers — a stand-in for the")
    print(" trained RL policy's decisions, which will be substituted in Week 9)")
    print("=" * 78)

    gen = BacklogGenerator(seed=42)
    backlog = gen.generate_backlog(8)
    capacity = 15

    tiers, ranked = assign_tiers(backlog, wsjf_score, capacity)

    for item in backlog:
        tier = tiers[item.item_id]
        explanation = explain_decision(item, tier)
        print(f"\n[{tier.name}] {item.name}")
        print(f"  -> {explanation}")
