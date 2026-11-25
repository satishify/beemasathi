"""Rule-based claim rejection predictor."""

from typing import Dict, List
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLAIM_RULES, INSURERS


class ClaimPredictor:
    """
    Predicts claim rejection risk based on hard-coded rules
    derived from IRDAI guidelines and common policy patterns.
    """

    def __init__(self):
        self.rules = CLAIM_RULES

    def predict(
        self,
        insurer: str,
        policy_name: str,
        hospital_name: str = None,
        is_network_hospital: bool = None,
        procedure: str = None,
        room_rent_per_day: float = None,
        sum_insured: float = None,
        has_ped: bool = None,
        months_since_policy: int = None,
        disclosed_conditions: List[str] = None,
        intimation_hours: int = None,
    ) -> Dict:
        """
        Returns rejection risk score (0-100) and actionable advice.
        """
        failed_checks = []
        passed_checks = []

        # Check 1: Network hospital
        if is_network_hospital is not None:
            if is_network_hospital:
                passed_checks.append("network_hospital")
            else:
                failed_checks.append({
                    "check": "network_hospital",
                    "weight": self.rules["network_hospital"]["rejection_weight"],
                    "message": f"{hospital_name or 'This hospital'} is NOT in "
                               f"{INSURERS.get(insurer, {}).get('name', insurer)}'s cashless network. "
                               "Reimbursement claim only. Rejection risk: HIGH if documents incomplete.",
                    "action": "Get pre-auth for reimbursement OR switch to network hospital."
                })

        # Check 2: Room rent cap
        if room_rent_per_day is not None and sum_insured is not None:
            cap = sum_insured * 0.01  # 1% is standard
            if room_rent_per_day > cap:
                failed_checks.append({
                    "check": "room_rent_within_cap",
                    "weight": self.rules["room_rent_within_cap"]["rejection_weight"],
                    "message": f"Room rent ₹{room_rent_per_day}/day exceeds cap of ₹{cap:.0f}/day (1% of SI). "
                               "All charges will be proportionately reduced.",
                    "action": f"Choose room ≤ ₹{cap:.0f}/day OR upgrade policy with no room rent cap."
                })
            else:
                passed_checks.append("room_rent_within_cap")

        # Check 3: PED waiting period
        if has_ped and months_since_policy is not None:
            if months_since_policy < 36:
                failed_checks.append({
                    "check": "ped_waiting_period",
                    "weight": self.rules["ped_waiting_period"]["rejection_weight"],
                    "message": f"PED waiting period: {months_since_policy}/36 months completed. "
                               "Claim for PED-related treatment will likely be REJECTED.",
                    "action": "Wait until 36 months OR claim only for unrelated condition."
                })
            else:
                passed_checks.append("ped_waiting_period")

        # Check 4: Cashless intimation
        if intimation_hours is not None:
            if intimation_hours > 24:
                failed_checks.append({
                    "check": "cashless_intimation_24h",
                    "weight": self.rules["cashless_intimation_24h"]["rejection_weight"],
                    "message": f"Intimation delayed by {intimation_hours} hours (>24h limit).",
                    "action": "File reimbursement claim with justification for delay."
                })
            else:
                passed_checks.append("cashless_intimation_24h")

        # Calculate risk score
        total_weight = sum(c["weight"] for c in failed_checks)
        risk_score = min(total_weight, 100)

        risk_level = "LOW"
        if risk_score > 30:
            risk_level = "MEDIUM"
        if risk_score > 60:
            risk_level = "HIGH"
        if risk_score > 85:
            risk_level = "VERY HIGH — EXPECT REJECTION"

        return {
            "rejection_risk_percent": risk_score,
            "risk_level": risk_level,
            "failed_checks": failed_checks,
            "passed_checks": passed_checks,
            "documents_needed": self._get_documents(procedure),
            "summary": self._build_summary(risk_score, failed_checks)
        }

    def _get_documents(self, procedure: str = None) -> List[str]:
        """Standard documents for health insurance claim."""
        docs = [
            "Discharge summary (original, signed by doctor)",
            "All original bills and receipts",
            "Investigation reports (X-ray, MRI, blood tests)",
            "Prescription from treating doctor",
            "Claim form (duly filled and signed)",
            "ID proof and policy copy",
            "Bank account details for NEFT",
        ]
        if procedure:
            docs.append(f"Procedure-specific: {procedure} consent form")
        return docs

    def _build_summary(self, risk_score: int, failed_checks: List[Dict]) -> str:
        if risk_score == 0:
            return "All basic checks passed. Ensure documents are complete before filing."

        reasons = "; ".join([c["message"].rstrip(".") for c in failed_checks[:2]])
        return f"Rejection risk: {risk_score}%. Main issues: {reasons}."


if __name__ == "__main__":
    predictor = ClaimPredictor()

    result = predictor.predict(
        insurer="star_health",
        policy_name="comprehensive",
        hospital_name="Apollo Chennai",
        is_network_hospital=False,
        room_rent_per_day=8000,
        sum_insured=500000,
        has_ped=True,
        months_since_policy=12,
        intimation_hours=48,
    )

    print(f"Rejection Risk: {result['rejection_risk_percent']}% ({result['risk_level']})")
    print(f"\nSummary: {result['summary']}")
    print(f"\nFailed Checks:")
    for c in result["failed_checks"]:
        print(f"  - {c['check']}: {c['message']}")
        print(f"    Action: {c['action']}")
