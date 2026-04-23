"""
JSON feature keys for POST /predict — must match the columns used when training `model_hmda.pkl`.

The bootstrap trainer (`bootstrap_hmda_model.py`) uses the same schema as `synthetic_hmda_rows`
after leakage columns are removed.
"""

REQUIRED_FIELDS = [
    "loan_amount_000s",
    "applicant_income_000s",
    "loan_type",
    "property_type",
    "applicant_ethnicity",
    "applicant_race_1",
    "applicant_sex",
    "hoepa_status",
    "lien_status",
    "population",
    "minority_population",
    "state_name",
    "loan_type_name",
    "applicant_ethnicity_name",
    "applicant_race_name_1",
]

# Counterfactual fairness check flips this HMDA field (integer codes: 1 = Male, 2 = Female typical)
COUNTERFACTUAL_ATTRIBUTE = "applicant_sex"

# Integer-coded HMDA fields (form submits as numbers)
INTEGER_FIELDS = {
    "loan_type",
    "property_type",
    "applicant_ethnicity",
    "applicant_race_1",
    "applicant_sex",
    "hoepa_status",
    "lien_status",
}

# Continuous / amount fields (JSON or forms may send these as strings)
FLOAT_FIELDS = (
    "loan_amount_000s",
    "applicant_income_000s",
    "population",
    "minority_population",
)

# UI: (field_name, input_type, label, section)
CUSTOMER_FORM_FIELDS: list[tuple[str, str, str, str]] = [
    ("loan_amount_000s", "number", "Loan amount ($000s)", "Loan"),
    ("applicant_income_000s", "number", "Applicant income ($000s)", "Applicant"),
    ("loan_type", "select", "Loan type", "Loan"),
    ("property_type", "select", "Property type", "Loan"),
    ("loan_type_name", "select", "Loan type (name)", "Loan"),
    ("applicant_ethnicity", "select", "Applicant ethnicity", "Applicant"),
    ("applicant_ethnicity_name", "select", "Applicant ethnicity (name)", "Applicant"),
    ("applicant_race_1", "select", "Applicant race (primary)", "Applicant"),
    ("applicant_race_name_1", "select", "Applicant race (name)", "Applicant"),
    ("applicant_sex", "select", "Applicant sex", "Applicant"),
    ("hoepa_status", "select", "HOEPA status", "Loan"),
    ("lien_status", "select", "Lien status", "Loan"),
    ("population", "number", "Tract population", "Census"),
    ("minority_population", "number", "Minority population (%)", "Census"),
    ("state_name", "select", "State (abbrev.)", "Property"),
]

# HMDA-style LAR codes (2017-era; aligns with public LAR code lists). Values are submitted as strings.
FIELD_SELECT_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "loan_type": [
        ("1", "1 — Conventional"),
        ("2", "2 — FHA-insured"),
        ("3", "3 — VA-guaranteed"),
        ("4", "4 — RHS/FSA"),
    ],
    "property_type": [
        ("1", "1 — One-to-four-family (other than manufactured)"),
        ("2", "2 — Manufactured housing"),
        ("3", "3 — Multifamily dwelling"),
    ],
    "loan_type_name": [
        ("Conventional", "Conventional"),
        ("FHA-insured", "FHA-insured"),
        ("VA-guaranteed", "VA-guaranteed"),
        ("RHS/FSA-guaranteed", "RHS/FSA-guaranteed"),
    ],
    "applicant_ethnicity": [
        ("1", "1 — Hispanic or Latino"),
        ("2", "2 — Not Hispanic or Latino"),
        ("3", "3 — Information not provided"),
        ("4", "4 — Not applicable"),
    ],
    "applicant_ethnicity_name": [
        ("Hispanic or Latino", "Hispanic or Latino"),
        ("Not Hispanic or Latino", "Not Hispanic or Latino"),
        ("Information not provided by applicant in mail, internet, or telephone application", "Information not provided"),
        ("Not applicable", "Not applicable"),
    ],
    "applicant_race_1": [
        ("1", "1 — American Indian or Alaska Native"),
        ("2", "2 — Asian"),
        ("3", "3 — Black or African American"),
        ("4", "4 — Native Hawaiian or Other Pacific Islander"),
        ("5", "5 — White"),
        ("6", "6 — Information not provided"),
        ("7", "7 — Not applicable"),
        ("8", "8 — No co-applicant"),
    ],
    "applicant_race_name_1": [
        ("American Indian or Alaska Native", "American Indian or Alaska Native"),
        ("Asian", "Asian"),
        ("Black or African American", "Black or African American"),
        ("Native Hawaiian or Other Pacific Islander", "Native Hawaiian or Other Pacific Islander"),
        ("White", "White"),
        ("Information not provided by applicant in mail, internet, or telephone application", "Information not provided"),
        ("Not applicable", "Not applicable"),
    ],
    "applicant_sex": [
        ("1", "1 — Male"),
        ("2", "2 — Female"),
        ("3", "3 — Joint (male/female)"),
        ("4", "4 — Applicant not applicable"),
        ("5", "5 — No co-applicant / not applicable"),
    ],
    "hoepa_status": [
        ("1", "1 — HOEPA loan"),
        ("2", "2 — Not a HOEPA loan"),
    ],
    "lien_status": [
        ("1", "1 — Secured by a first lien"),
        ("2", "2 — Secured by a subordinate lien"),
        ("3", "3 — Not secured by a lien"),
        ("4", "4 — Exempt (not applicable)"),
    ],
    "state_name": [
        ("AL", "Alabama (AL)"),
        ("AK", "Alaska (AK)"),
        ("AZ", "Arizona (AZ)"),
        ("AR", "Arkansas (AR)"),
        ("CA", "California (CA)"),
        ("CO", "Colorado (CO)"),
        ("CT", "Connecticut (CT)"),
        ("DE", "Delaware (DE)"),
        ("DC", "District of Columbia (DC)"),
        ("FL", "Florida (FL)"),
        ("GA", "Georgia (GA)"),
        ("HI", "Hawaii (HI)"),
        ("ID", "Idaho (ID)"),
        ("IL", "Illinois (IL)"),
        ("IN", "Indiana (IN)"),
        ("IA", "Iowa (IA)"),
        ("KS", "Kansas (KS)"),
        ("KY", "Kentucky (KY)"),
        ("LA", "Louisiana (LA)"),
        ("ME", "Maine (ME)"),
        ("MD", "Maryland (MD)"),
        ("MA", "Massachusetts (MA)"),
        ("MI", "Michigan (MI)"),
        ("MN", "Minnesota (MN)"),
        ("MS", "Mississippi (MS)"),
        ("MO", "Missouri (MO)"),
        ("MT", "Montana (MT)"),
        ("NE", "Nebraska (NE)"),
        ("NV", "Nevada (NV)"),
        ("NH", "New Hampshire (NH)"),
        ("NJ", "New Jersey (NJ)"),
        ("NM", "New Mexico (NM)"),
        ("NY", "New York (NY)"),
        ("NC", "North Carolina (NC)"),
        ("ND", "North Dakota (ND)"),
        ("OH", "Ohio (OH)"),
        ("OK", "Oklahoma (OK)"),
        ("OR", "Oregon (OR)"),
        ("PA", "Pennsylvania (PA)"),
        ("RI", "Rhode Island (RI)"),
        ("SC", "South Carolina (SC)"),
        ("SD", "South Dakota (SD)"),
        ("TN", "Tennessee (TN)"),
        ("TX", "Texas (TX)"),
        ("UT", "Utah (UT)"),
        ("VT", "Vermont (VT)"),
        ("VA", "Virginia (VA)"),
        ("WA", "Washington (WA)"),
        ("WV", "West Virginia (WV)"),
        ("WI", "Wisconsin (WI)"),
        ("WY", "Wyoming (WY)"),
    ],
}
