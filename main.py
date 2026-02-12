import joblib
import os
import pandas as pd


class LoanApprovalApp:

    def __init__(self,
                 classifier='stage_1_rf_classifier_pipeline.pkl',
                 regressor="stage_2_rf_regression_pipeline.pkl"):

        # Get absolute path (SAFE way)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "models")

        try:
            self.clf = joblib.load(os.path.join(model_path, classifier))
            self.reg = joblib.load(os.path.join(model_path, regressor))
        except Exception as e:
            raise RuntimeError(f"❌ Model loading failed: {e}")

    # ---------------------------------------------------

    def get_user_input(self):

        print("--- Using Sample Applicant Data ---\n")

        # ⚠️ Use realistic financial values
        data = {
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 850000,
            "loan_amount": 250000,
            "loan_term": 60,
            "cibil_score": 750,   # FIXED (was 4 ❌)
            "residential_assets_value": 1200000,
            "commercial_assets_value": 400000,
            "luxury_assets_value": 150000,
            "bank_asset_value": 300000
        }

        return pd.DataFrame([data])

    # ---------------------------------------------------

    def two_stage_predict(self, applicant_df):

        out = {}

        # Align columns with training schema
        applicant_df = applicant_df[self.clf.feature_names_in_]

        # Safety check
        if applicant_df.isnull().any().any():
            raise ValueError("❌ Input contains missing values!")

        # ⭐ Get probability (much better than raw label)
        prob = self.clf.predict_proba(applicant_df)[0][1]
        approve = prob > 0.60   # safer threshold

        out["approve"] = int(approve)
        out["probability"] = float(prob)

        if approve:
            applicant_df_reg = applicant_df.copy()
            applicant_df_reg['loan_status'] = 'Approve'

            pred = self.reg.predict(applicant_df_reg)[0]
            out["regression_prediction"] = float(pred)
        else:
            out["regression_prediction"] = None

        return out

    # ---------------------------------------------------

    def run(self):

        print("\n============================")
        print("   Loan Approval System")
        print("============================\n")

        applicant_df = self.get_user_input()
        result = self.two_stage_predict(applicant_df)

        print("------ RESULT ------")

        print(f"Approval Probability : {result['probability']:.2%}")

        if result["approve"]:
            print("Loan Status : ✅ APPROVED")
            print(f"Recommended Loan Value : {result['regression_prediction']:.2f}")
        else:
            print("Loan Status : ❌ REJECTED")

        print("\n---------------------\n")

        return result


# ---------------------------------------------------

if __name__ == "__main__":
    app = LoanApprovalApp()
    app.run()
