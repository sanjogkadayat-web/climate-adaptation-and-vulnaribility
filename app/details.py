import pandas as pd

# column names in country_scored.csv
NAME, RESID, INCOME, REGION = "country", "misallocation_mean", "income_tier", "region"
NYEARS, PROFILE, THIN = "n_years", "profile", "thin_data"
VULN, TIER_RANK, TIER_N = "vulnerability_mean", "tier_rank", "tier_n"

TIER_LABEL = {"H": "high-income", "UM": "upper-middle-income",
              "LM": "lower-middle-income", "L": "low-income"}


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th")
    return f"{n}{suffix}"


def country_summary(df: pd.DataFrame, country: str):
    """Return (metrics dict, country_markdown, region_markdown) for one country."""
    row = df[df[NAME] == country].iloc[0]
    region = row[REGION]
    resid = float(row[RESID])
    tier = TIER_LABEL.get(row[INCOME], row[INCOME])
    rank, n = int(row[TIER_RANK]), int(row[TIER_N])

    extreme = ""
    if rank == 1:
        extreme = " (the most underfunded in its tier)"
    elif rank == n:
        extreme = " (the most over-resourced in its tier)"
    direction = "below" if resid < 0 else "above"

    thin_note = " Based on only one or two years, so read with caution." if bool(row[THIN]) else ""
    country_md = (
        f"**{country}** is **{row[PROFILE]}**. Its adaptation aid sits "
        f"{abs(resid):.2f} log units {direction} the model's prediction, ranking "
        f"{_ordinal(rank)} of {n} in the {tier} tier{extreme}. "
        f"Average vulnerability over {int(row[NYEARS])} years is {row[VULN]:.2f}.{thin_note}"
    )

    reg = df[df[REGION] == region]
    N = len(reg)
    rmean = reg[RESID].mean()
    rdir = "under-allocated" if rmean < 0 else "over-allocated"
    if N == 1:
        region_md = (
            f"{country} is the only sampled country in **{region}**, "
            "so no regional comparison is available."
        )
    else:
        gap = resid - rmean
        rel = "more underfunded than" if gap < 0 else "better funded than"
        share = (reg[RESID] < 0).mean()
        share_txt = f" {share:.0%} of them fall below the model line, and" if N >= 5 else ""
        region_md = (
            f"Across the {N} sampled countries in **{region}**, the average "
            f"misallocation is {rmean:+.2f} ({rdir} overall).{share_txt} {country} runs "
            f"{abs(gap):.2f} log units {rel} that regional average."
        )

    metrics = {"resid": resid, "rank": rank, "tier_n": n,
               "vuln": float(row[VULN]), "profile": row[PROFILE]}
    return metrics, country_md, region_md