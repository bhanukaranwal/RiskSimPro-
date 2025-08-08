import datetime

class BaselIIIReport:
    def __init__(self, portfolio_data, capital_requirements, risk_metrics):
        self.portfolio_data = portfolio_data
        self.capital_requirements = capital_requirements
        self.risk_metrics = risk_metrics
        self.generated_date = datetime.datetime.now()

    def generate_report(self):
        # Generate a detailed Basel III compliance report structure
        report = {
            "report_name": "Basel III Compliance Report",
            "generated_on": self.generated_date.isoformat(),
            "portfolio_summary": self.portfolio_data,
            "capital_requirements": self.capital_requirements,
            "risk_metrics": self.risk_metrics,
            "key_ratios": self.calculate_key_ratios(),
            "compliance_status": self.check_compliance()
        }
        return report

    def calculate_key_ratios(self):
        # Example calculations for core capital ratios (CET1, Tier 1 Capital, Total Capital)
        total_rwa = self.risk_metrics.get("risk_weighted_assets", 1.0)
        cet1 = self.capital_requirements.get("CET1", 0)
        tier1 = self.capital_requirements.get("Tier1", 0)
        total_capital = self.capital_requirements.get("TotalCapital", 0)

        return {
            "CET1_ratio": cet1 / total_rwa if total_rwa > 0 else 0,
            "Tier1_ratio": tier1 / total_rwa if total_rwa > 0 else 0,
            "Total_capital_ratio": total_capital / total_rwa if total_rwa > 0 else 0
        }

    def check_compliance(self):
        key_ratios = self.calculate_key_ratios()
        cet1_req = 0.04  # Minimum 4% CET1
        tier1_req = 0.06  # Minimum 6% Tier 1 Capital
        total_req = 0.08  # Minimum 8% Total Capital

        compliance = {
            "CET1_compliant": key_ratios["CET1_ratio"] >= cet1_req,
            "Tier1_compliant": key_ratios["Tier1_ratio"] >= tier1_req,
            "Total_capital_compliant": key_ratios["Total_capital_ratio"] >= total_req
        }
        return compliance

class DoddFrankReport:
    def __init__(self, portfolio_data, stress_test_results, required_disclosures):
        self.portfolio_data = portfolio_data
        self.stress_test_results = stress_test_results
        self.required_disclosures = required_disclosures
        self.generated_date = datetime.datetime.now()

    def generate_report(self):
        report = {
            "report_name": "Dodd-Frank Compliance Report",
            "generated_on": self.generated_date.isoformat(),
            "portfolio_summary": self.portfolio_data,
            "stress_test_results": self.stress_test_results,
            "disclosures": self.required_disclosures,
            "compliance_checks": self.check_compliance()
        }
        return report

    def check_compliance(self):
        # Placeholder: compliance checks matching Dodd-Frank stipulations.
        # Example: Verify required stress tests have been performed and limits not breached
        stress_pass = all(result.get("passed", True) for result in self.stress_test_results.values())

        disclosure_required_fields = ["counterparty_risk", "market_risk", "operational_risk"]
        disclosures_complete = all(field in self.required_disclosures for field in disclosure_required_fields)

        return {
            "stress_tests_passed": stress_pass,
            "required_disclosures_present": disclosures_complete
        }

class SolvencyIIReport:
    def __init__(self, portfolio_data, solvency_capital_requirement, minimum_capital_requirement):
        self.portfolio_data = portfolio_data
        self.scr = solvency_capital_requirement
        self.mcr = minimum_capital_requirement
        self.generated_date = datetime.datetime.now()

    def generate_report(self):
        report = {
            "report_name": "Solvency II Compliance Report",
            "generated_on": self.generated_date.isoformat(),
            "portfolio_summary": self.portfolio_data,
            "SCR": self.scr,
            "MCR": self.mcr,
            "compliance_status": self.check_compliance()
        }
        return report

    def check_compliance(self):
        scr_threshold = 1.0
        mcr_threshold = 1.0

        scr_compliant = self.scr >= scr_threshold
        mcr_compliant = self.mcr >= mcr_threshold

        return {
            "SCR_compliant": scr_compliant,
            "MCR_compliant": mcr_compliant
        }
