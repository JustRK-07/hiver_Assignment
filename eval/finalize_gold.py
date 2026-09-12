"""Append g051–g150 onto eval/golden_set.csv from the candidate worksheet.

Each extra row was read as a tweet (not auto-accepted from suggested_intent).
Rules match g001–g050: one action-intent; escalate for human/legal/fraud/self-harm
or when a correct auto-reply would need a PNR / invented refund.

Re-run:
    PYTHONPATH=. python eval/finalize_gold.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from agent.config import ROOT

CANDIDATES = ROOT / "eval" / "golden_set_candidates.csv"
GOLD = ROOT / "eval" / "golden_set.csv"
PAIRS = ROOT / "data" / "pairs_cluster.csv"

# customer_tweet_id -> (gold_intent, gold_escalate, reason, notes)
# escalate reasons must match agent.gate rule names when Tier-1 should fire.
LABELS: dict[str, tuple[str, str, str, str]] = {
    # --- delays ---
    "1517255": ("flight_delay", "false", "", "multi: delay + catering; action is the delay"),
    "1590121": ("flight_delay", "false", "", "repeat delay anxiety; operational delay"),
    "966454": ("flight_delay", "false", "", "short-haul delay, no status updates"),
    "5037": ("flight_delay", "false", "", "delay + missing ETA"),
    "1987610": ("flight_delay", "false", "", "punctuality distrust after a 26h delay"),
    "2665014": ("flight_delay", "false", "", "3h wait then food; delay is the event"),
    "2003228": ("flight_delay", "false", "", "multi: weather delay + check-in + seats"),
    # --- cancel / rebook ---
    "648106": ("cancellation_rebook", "true", "request_human", "cancelled + cannot reach phone; needs a human"),
    "367109": ("cancellation_rebook", "false", "", "schedule posted then cancelled"),
    "1643428": ("cancellation_rebook", "false", "", "cancelled with no notification at airport"),
    "666752": ("cancellation_rebook", "false", "", "cancelled; stranded in hotel queue"),
    "1644887": ("cancellation_rebook", "false", "", "knock-on hire-car after cancel"),
    "2877202": ("cancellation_rebook", "false", "", "name change only via cancel-and-rebook"),
    "1502231": ("cancellation_rebook", "false", "", "missed connection; wants rebook"),
    "933485": ("cancellation_rebook", "false", "", "already rebooked; gratitude, still cancel class"),
    # --- baggage ---
    "349486": ("baggage", "false", "", "missing bag, known last scan ORD"),
    "121758": ("baggage", "false", "", "repeat lost suitcase"),
    "292461": ("baggage", "false", "", "lost property with high-value items"),
    "2610536": ("baggage", "false", "", "checked-bag allowance FAQ"),
    "781344": ("baggage", "false", "", "allowance on a specific PNR — FAQ-shaped"),
    "1904338": ("baggage", "false", "", "cabin item (penny board) policy"),
    "2008391": ("baggage", "false", "", "missing pram four days after arrival"),
    "482195": ("baggage", "false", "", "sarcastic thanks for lost bags"),
    # --- refund / compensation ---
    "825261": ("refund_compensation", "false", "", "no refunds after hurricane disruption"),
    "1447491": ("refund_compensation", "false", "", "taxi cost; wants to file a claim"),
    "2225222": ("refund_compensation", "false", "", "EU261 disputed landing time"),
    "2493423": ("refund_compensation", "false", "", "EU261 form error, wants help"),
    "369335": ("refund_compensation", "false", "", "paltry £50 after poor club service"),
    "848834": ("refund_compensation", "false", "", "misinfo led to bus journey; no payout"),
    "230479": ("refund_compensation", "false", "", "chasing an already-requested refund"),
    "1618806": ("refund_compensation", "false", "", "waiting on CS for a refund"),
    # --- booking / check-in ---
    "544110": ("booking_checkin", "false", "", "website booking friction"),
    "291941": ("booking_checkin", "false", "", "online check-in unavailable"),
    "2810678": ("booking_checkin", "false", "", "check-in rejects complete passport data"),
    "8132": ("booking_checkin", "false", "", "website broken"),
    "439344": ("booking_checkin", "true", "ambiguous_needs_human", "emergency date change + MMB down"),
    "1420304": ("booking_checkin", "false", "", "cannot change flight, systems error"),
    "365750": ("booking_checkin", "false", "", "third-party helper error blocking date change"),
    "347038": ("booking_checkin", "false", "", "desk opening time FAQ"),
    # --- seats / upgrades ---
    "366249": ("seats_upgrade", "false", "", "paid seat map empty in app"),
    "2785980": ("seats_upgrade", "false", "", "charged to reserve a club seat"),
    "519600": ("seats_upgrade", "false", "", "pay-to-upgrade an Avios reward"),
    "1844269": ("seats_upgrade", "false", "", "asks complimentary upgrade — do not invent"),
    "548529": ("seats_upgrade", "true", "ambiguous_needs_human", "paid front seats, assigned row 26, claustrophobia"),
    "1918939": ("seats_upgrade", "false", "", "twin-seat product FAQ"),
    "1986292": ("seats_upgrade", "false", "", "bronze free-seat window not honoured"),
    "914966": ("seats_upgrade", "false", "", "selected seat keeps changing"),
    # --- loyalty / Avios ---
    "36191": ("loyalty_avios", "false", "", "exec club / combine Avios contact path"),
    "2049190": ("loyalty_avios", "false", "", "add companion frequent-flyer after travel"),
    "886516": ("loyalty_avios", "false", "", "leave exec club and take Avios"),
    "2035003": ("loyalty_avios", "false", "", "pool two members' Avios for a hotel"),
    "2530433": ("loyalty_avios", "false", "", "missing silver points on intl flights"),
    "1339978": ("loyalty_avios", "false", "", "price-match ask as silver — do not invent"),
    "2569329": ("loyalty_avios", "false", "", "silver CS complaint + lost bookings"),
    "385992": ("loyalty_avios", "false", "", "agency ticket number needed to reclaim Avios"),
    # --- inflight product ---
    "1739349": ("inflight_product", "false", "", "meal looks good, no taste"),
    "164473": ("inflight_product", "false", "", "WTP wifi/power FAQ"),
    "2611424": ("inflight_product", "false", "", "no usable gluten-free breakfast"),
    "1056708": ("inflight_product", "false", "", "GF meal missing despite SSR"),
    "2838968": ("inflight_product", "false", "", "A380 wifi FAQ"),
    "2174643": ("inflight_product", "true", "ambiguous_needs_human", "left notebook onboard — needs a trace"),
    "381324": ("inflight_product", "false", "", "broken recliner / cracked IFE; paltry rebate"),
    # --- airport / staff / lounge ---
    "1551482": ("airport_staff", "false", "", "Heathrow service failure, misconnect Madrid"),
    "1390432": ("airport_staff", "false", "", "lounge staff handed wine with no glass"),
    "2826609": ("airport_staff", "true", "ambiguous_needs_human", "iPad left in lounge — lost property"),
    "1451920": ("airport_staff", "false", "", "genuine staff praise EDI"),
    "2197499": ("airport_staff", "false", "", "genuine crew praise BA143"),
    "1045707": ("airport_staff", "false", "", "told misconnect an hour before departure"),
    "121727": ("airport_staff", "false", "", "uneven cabin-crew workload"),
    "2022730": ("airport_staff", "false", "", "rude / miserable cabin crew"),
    # --- explicit human / phone ---
    "1063702": ("refund_compensation", "true", "request_human", "wants a human; also owed taxi money"),
    "1124734": ("booking_checkin", "true", "request_human", "call me to fix a name change"),
    "1906925": ("booking_checkin", "true", "request_human", "cannot lodge complaint or reach phone"),
    "2412071": ("flight_delay", "true", "request_human", "misconnect; asks for a phone number"),
    "1106928": ("baggage", "true", "request_human", "lost bag + waiting for a manager"),
    "130666": ("general_query", "true", "request_human", "will continue only with a real person"),
    "2117536": ("general_query", "true", "request_human", "Chile CS phone number"),
    "660067": ("booking_checkin", "true", "request_human", "wants a free phone number vs website"),
    # --- legal / fraud ---
    "949359": ("loyalty_avios", "true", "fraud", "Avios hotel booked fraudulently"),
    "20831": ("refund_compensation", "true", "legal", "small-claims court vs refund now"),
    # --- sarcasm / praise-complaints ---
    "1536888": ("airport_staff", "false", "", "sarcastic 'brilliant CS' / jargon"),
    "1059990": ("airport_staff", "false", "", "sarcastic 'well done' on corporate BS"),
    "1728170": ("baggage", "false", "", "'brilliant flight' then missing luggage"),
    "1033959": ("airport_staff", "false", "", "'here to help' desk not actually helping"),
    "22458": ("loyalty_avios", "false", "", "sarcastic 'excellent job' still no silver fix"),
    "542823": ("airport_staff", "false", "", "lounge agent helpful after exec-club line failed"),
    # --- ambiguous / DM-only history ---
    "1920766": ("general_query", "true", "ambiguous_needs_human", "security bug bounty — do not auto-handle"),
    "2197496": ("booking_checkin", "false", "", "umlaut vs ue on the ticket — policy FAQ"),
    "983575": ("general_query", "false", "", "birthday treat ask — do not invent perks"),
    "1119503": ("general_query", "true", "ambiguous_needs_human", "customer wants a private channel"),
    "920611": ("booking_checkin", "false", "", "agent-made booking not on the account"),
    "2030858": ("booking_checkin", "false", "", "cannot find booking reference"),
    "742626": ("booking_checkin", "true", "ambiguous_needs_human", "hour wait, no booking records showing"),
    "1538306": ("seats_upgrade", "true", "request_human", "cannot change seats online; wants CS"),
    # --- leftover multi / general ---
    "1181812": ("baggage", "false", "", "split bag delivery + useless CS"),
    "326277": ("refund_compensation", "false", "", "driver wait fees vs a lost-bag claim"),
    "1046318": ("booking_checkin", "false", "", "forced airport check-in, no seat map"),
    "2708335": ("general_query", "false", "", "bedbox allowed on long-haul?"),
}


CONSTRUCTED = [
    {
        "id": "g149",
        "customer_text": (
            "My solicitor will file a lawsuit tomorrow if this refund is not processed. "
            "I already have a lawyer on retainer."
        ),
        "gold_intent": "refund_compensation",
        "gold_escalate": "true",
        "gold_escalate_reason": "legal",
        "notes": "constructed legal edge (rare in BA subsample); solicitor + lawsuit",
        "relevant_company_tweet_id": "",
    },
    {
        "id": "g150",
        "customer_text": (
            "I am going to kill myself if you cancel this flight again. This holiday was "
            "the only thing I had left."
        ),
        "gold_intent": "cancellation_rebook",
        "gold_escalate": "true",
        "gold_escalate_reason": "self_harm",
        "notes": "constructed self-harm + cancel; Tier-1 must win over rebook drafting",
        "relevant_company_tweet_id": "",
    },
]


def load_by_id(path: Path, key: str) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {row[key]: row for row in csv.DictReader(f) if row.get(key)}


def main() -> None:
    existing = list(csv.DictReader(GOLD.open(encoding="utf-8")))
    # Keep the original 50; drop a previous expansion if this script is re-run.
    base = [r for r in existing if r["id"] < "g051"]
    cand = load_by_id(CANDIDATES, "customer_tweet_id")
    pairs = load_by_id(PAIRS, "customer_tweet_id")

    missing = [tid for tid in LABELS if tid not in cand]
    if missing:
        raise SystemExit(f"tweet ids not in candidates: {missing}")

    extra = []
    n = 51
    for tid, (intent, esc, reason, notes) in LABELS.items():
        row = cand[tid]
        company_id = (pairs.get(tid) or {}).get("company_tweet_id", "")
        extra.append(
            {
                "id": f"g{n:03d}",
                "customer_text": row["customer_text"],
                "gold_intent": intent,
                "gold_escalate": esc,
                "gold_escalate_reason": reason,
                "notes": notes,
                "relevant_company_tweet_id": company_id,
            }
        )
        n += 1
    extra.extend(CONSTRUCTED)
    if len(base) + len(extra) != 150:
        raise SystemExit(f"expected 150 rows, got {len(base)+len(extra)} (base={len(base)} extra={len(extra)})")

    fieldnames = [
        "id",
        "customer_text",
        "gold_intent",
        "gold_escalate",
        "gold_escalate_reason",
        "notes",
        "relevant_company_tweet_id",
    ]
    with GOLD.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in base:
            w.writerow({k: row.get(k, "") for k in fieldnames})
        w.writerows(extra)

    from collections import Counter

    all_rows = base + extra
    print("n=", len(all_rows))
    print("intents", Counter(r["gold_intent"] for r in all_rows))
    print("escalate", Counter(r["gold_escalate"] for r in all_rows))
    print("reasons", Counter(r["gold_escalate_reason"] for r in all_rows if r["gold_escalate"] == "true"))


if __name__ == "__main__":
    main()
