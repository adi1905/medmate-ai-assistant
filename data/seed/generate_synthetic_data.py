"""
MedMate synthetic data generator
---------------------------------
Builds a realistic-looking SQLite database matching the MedMate schema,
using Faker + curated real-world medical reference data (drug classes,
lab panels with clinically plausible reference ranges, specialties, etc.)

Run:
    python3 generate_medmate_data.py

Produces:
    medmate.db   (SQLite database, ready to query)
"""

import sqlite3
import random
from datetime import datetime, timedelta
from faker import Faker
from config.settings import DB_PATH, SCHEMA_PATH

random.seed(42)
fake = Faker()
Faker.seed(42)

# ------------------------------------------------------------------
# Row-count targets (tuned to look like a real mid-size clinic)
# ------------------------------------------------------------------
N_PROVIDERS = 28
N_PATIENTS = 650
N_APPOINTMENTS = 3800
N_PRESCRIPTIONS = 2100
N_LAB_RESULTS = 5400
N_MEMORIES = 950

now = datetime(2026, 7, 15, 9, 0, 0)


def iso_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def iso_date(dt):
    return dt.strftime("%Y-%m-%d")


def random_past_datetime(days_back_max, days_back_min=0):
    delta = random.randint(days_back_min, days_back_max)
    seconds = random.randint(0, 86399)
    return now - timedelta(days=delta, seconds=-seconds if seconds < 0 else 0) - timedelta(seconds=random.randint(0, 86399))


def random_dt_between(start, end):
    delta = end - start
    secs = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=secs)


# ==================================================================
# 1. PROVIDERS
# ==================================================================
SPECIALTIES = [
    "Family Medicine", "Internal Medicine", "Cardiology", "Endocrinology",
    "Pediatrics", "Dermatology", "Orthopedics", "Psychiatry", "Neurology",
    "Obstetrics & Gynecology", "Gastroenterology", "Pulmonology",
    "Rheumatology", "Nephrology", "Urology", "Otolaryngology",
]

TITLES = ["Dr.", "Dr.", "Dr.", "NP", "PA"]  # mostly MDs, some NPs/PAs

providers = []
for i in range(1, N_PROVIDERS + 1):
    pid = f"PR-{i:03d}"
    title = random.choice(TITLES)
    name = f"{title} {fake.first_name()} {fake.last_name()}"
    specialty = random.choice(SPECIALTIES)
    created = random_past_datetime(days_back_max=2500, days_back_min=200)
    providers.append((pid, name, specialty, iso_dt(created)))

# ==================================================================
# 2. MEDICATIONS REFERENCE (curated, clinically-plausible)
# ==================================================================
medications_reference = [
    ("lisinopril", "ACE Inhibitor", "10mg once daily", "Monitor potassium and renal function; may cause dry cough."),
    ("amlodipine", "Calcium Channel Blocker", "5mg once daily", "Watch for peripheral edema."),
    ("metoprolol succinate", "Beta Blocker", "50mg once daily", "Do not discontinue abruptly."),
    ("losartan", "ARB", "50mg once daily", "Alternative to ACE inhibitors if cough develops."),
    ("hydrochlorothiazide", "Thiazide Diuretic", "25mg once daily", "Monitor sodium and potassium levels."),
    ("atorvastatin", "Statin", "20mg once daily at bedtime", "Monitor liver enzymes; watch for myalgia."),
    ("rosuvastatin", "Statin", "10mg once daily", "Similar monitoring to other statins."),
    ("simvastatin", "Statin", "40mg once daily at bedtime", "Avoid grapefruit juice."),
    ("metformin", "Biguanide (Antidiabetic)", "500mg twice daily", "Take with food; monitor renal function."),
    ("glipizide", "Sulfonylurea", "5mg once daily before breakfast", "Risk of hypoglycemia."),
    ("insulin glargine", "Long-acting Insulin", "10 units subcutaneous at bedtime", "Titrate based on fasting glucose."),
    ("sitagliptin", "DPP-4 Inhibitor", "100mg once daily", "Generally well tolerated."),
    ("empagliflozin", "SGLT2 Inhibitor", "10mg once daily", "Monitor for genital infections and dehydration."),
    ("levothyroxine", "Thyroid Hormone", "75mcg once daily", "Take on empty stomach; recheck TSH in 6-8 weeks after dose change."),
    ("albuterol", "Short-acting Beta Agonist (Bronchodilator)", "2 puffs every 4-6 hours as needed", "Rescue inhaler for bronchospasm."),
    ("fluticasone propionate", "Inhaled Corticosteroid", "1-2 sprays each nostril daily", "For allergic rhinitis."),
    ("montelukast", "Leukotriene Receptor Antagonist", "10mg once daily at bedtime", "Monitor for neuropsychiatric effects."),
    ("omeprazole", "Proton Pump Inhibitor", "20mg once daily before breakfast", "Long-term use linked to reduced B12 absorption."),
    ("pantoprazole", "Proton Pump Inhibitor", "40mg once daily", "Similar profile to omeprazole."),
    ("ranitidine substitute famotidine", "H2 Blocker", "20mg twice daily", "Alternative to PPIs for GERD."),
    ("sertraline", "SSRI", "50mg once daily", "Allow 4-6 weeks for full effect; monitor mood changes."),
    ("escitalopram", "SSRI", "10mg once daily", "Similar to sertraline; fewer drug interactions."),
    ("fluoxetine", "SSRI", "20mg once daily", "Long half-life; less withdrawal risk."),
    ("bupropion", "Atypical Antidepressant", "150mg once daily", "Lowers seizure threshold; avoid in eating disorders."),
    ("venlafaxine", "SNRI", "75mg once daily", "Monitor blood pressure."),
    ("trazodone", "Serotonin Antagonist/Reuptake Inhibitor", "50mg at bedtime as needed for sleep", "Sedating; risk of priapism (rare)."),
    ("alprazolam", "Benzodiazepine", "0.25mg twice daily as needed", "Risk of dependence; short-term use preferred."),
    ("lorazepam", "Benzodiazepine", "0.5mg twice daily as needed", "Caution with CNS depressants."),
    ("gabapentin", "Anticonvulsant / Neuropathic Pain Agent", "300mg three times daily", "Titrate slowly; renal dose adjustment needed."),
    ("pregabalin", "Anticonvulsant / Neuropathic Pain Agent", "75mg twice daily", "Similar uses to gabapentin."),
    ("duloxetine", "SNRI", "60mg once daily", "Also used for neuropathic pain and fibromyalgia."),
    ("amoxicillin", "Penicillin Antibiotic", "500mg three times daily for 7 days", "Check for penicillin allergy before prescribing."),
    ("amoxicillin-clavulanate", "Penicillin Antibiotic", "875mg twice daily for 7 days", "GI upset common; take with food."),
    ("azithromycin", "Macrolide Antibiotic", "500mg day 1, then 250mg daily for 4 days", "Watch for QT prolongation."),
    ("ciprofloxacin", "Fluoroquinolone Antibiotic", "500mg twice daily for 7 days", "Avoid dairy/antacids near dosing; tendon rupture risk."),
    ("doxycycline", "Tetracycline Antibiotic", "100mg twice daily for 10 days", "Photosensitivity; avoid in pregnancy."),
    ("cephalexin", "Cephalosporin Antibiotic", "500mg four times daily for 7 days", "Cross-reactivity risk with penicillin allergy."),
    ("nitrofurantoin", "Urinary Antiseptic", "100mg twice daily for 5 days", "Used for uncomplicated UTIs; avoid in renal impairment."),
    ("prednisone", "Corticosteroid", "20mg once daily, taper over 7 days", "Take with food; taper to avoid adrenal suppression."),
    ("ibuprofen", "NSAID", "400mg every 6-8 hours as needed", "Take with food; monitor renal function with long-term use."),
    ("naproxen", "NSAID", "500mg twice daily", "GI protection recommended with long-term use."),
    ("acetaminophen", "Analgesic/Antipyretic", "500mg every 6 hours as needed", "Max 3000mg/day; hepatotoxic in overdose."),
    ("tramadol", "Opioid Analgesic (weak)", "50mg every 6 hours as needed", "Lowers seizure threshold; monitor for dependence."),
    ("cyclobenzaprine", "Muscle Relaxant", "10mg three times daily as needed", "Sedating; short-term use for muscle spasm."),
    ("warfarin", "Anticoagulant", "5mg once daily, adjust per INR", "Requires regular INR monitoring."),
    ("apixaban", "Direct Oral Anticoagulant", "5mg twice daily", "No routine monitoring required; renal dose adjustment."),
    ("clopidogrel", "Antiplatelet", "75mg once daily", "Often combined with aspirin post-stent."),
    ("aspirin low-dose", "Antiplatelet", "81mg once daily", "Cardioprotective dose; monitor for GI bleeding."),
    ("furosemide", "Loop Diuretic", "40mg once daily", "Monitor potassium and renal function."),
    ("spironolactone", "Potassium-sparing Diuretic", "25mg once daily", "Monitor potassium closely, especially with ACEi/ARB."),
    ("allopurinol", "Xanthine Oxidase Inhibitor", "300mg once daily", "Used for gout prophylaxis; not during acute flare."),
    ("colchicine", "Anti-gout Agent", "0.6mg twice daily", "GI side effects common; dose reduce in renal impairment."),
    ("methotrexate", "DMARD", "15mg once weekly", "Requires folic acid supplementation and LFT monitoring."),
    ("hydroxychloroquine", "DMARD", "200mg twice daily", "Requires periodic ophthalmologic exams."),
    ("levetiracetam", "Anticonvulsant", "500mg twice daily", "Monitor for mood/behavioral changes."),
    ("lamotrigine", "Anticonvulsant / Mood Stabilizer", "100mg twice daily", "Slow titration required to reduce rash risk."),
    ("sumatriptan", "Triptan (Migraine)", "50mg at onset, may repeat once after 2 hours", "Max 200mg/day; avoid in cardiovascular disease."),
    ("propranolol", "Beta Blocker", "40mg twice daily", "Also used for migraine prophylaxis and tremor."),
    ("tamsulosin", "Alpha Blocker", "0.4mg once daily", "Used for BPH; orthostatic hypotension risk."),
    ("finasteride", "5-alpha Reductase Inhibitor", "5mg once daily", "Used for BPH; category X in pregnancy exposure."),
    ("estradiol", "Hormone Replacement", "1mg once daily", "Assess VTE risk before starting."),
    ("norethindrone-ethinyl estradiol", "Oral Contraceptive", "1 tablet once daily", "Counsel on missed-dose protocol."),
    ("prenatal vitamin", "Supplement", "1 tablet once daily", "Contains folic acid, iron, and DHA."),
    ("ferrous sulfate", "Iron Supplement", "325mg once daily", "Take with vitamin C; may cause constipation."),
    ("vitamin D3", "Supplement", "2000 IU once daily", "Recheck 25-OH vitamin D level in 3 months."),
    ("cetirizine", "Antihistamine", "10mg once daily", "Non-sedating; used for allergic rhinitis."),
    ("loratadine", "Antihistamine", "10mg once daily", "Non-sedating antihistamine."),
    ("melatonin", "Sleep Aid (OTC)", "3mg at bedtime as needed", "Generally well tolerated."),
    ("zolpidem", "Sedative-Hypnotic", "5mg at bedtime as needed", "Short-term use only; risk of complex sleep behaviors."),
    ("clonazepam", "Benzodiazepine", "0.5mg twice daily", "Used for anxiety/seizures; taper to discontinue."),
    ("buspirone", "Anxiolytic (non-benzodiazepine)", "15mg twice daily", "Non-habit forming; takes 2-4 weeks for effect."),
]

# ==================================================================
# 3. LAB TEST PANELS (test_name, unit, ref_low, ref_high, value_generator)
# ==================================================================
LAB_TESTS = [
    ("Glucose, Fasting", "mg/dL", 70, 99, lambda: round(random.gauss(92, 22), 1)),
    ("Hemoglobin A1c", "%", 4.0, 5.6, lambda: round(random.gauss(5.6, 1.1), 1)),
    ("Total Cholesterol", "mg/dL", 125, 200, lambda: round(random.gauss(190, 40), 0)),
    ("LDL Cholesterol", "mg/dL", 0, 99, lambda: round(random.gauss(105, 35), 0)),
    ("HDL Cholesterol", "mg/dL", 40, 60, lambda: round(random.gauss(50, 15), 0)),
    ("Triglycerides", "mg/dL", 0, 149, lambda: round(random.gauss(130, 60), 0)),
    ("Sodium", "mmol/L", 135, 145, lambda: round(random.gauss(140, 3), 1)),
    ("Potassium", "mmol/L", 3.5, 5.1, lambda: round(random.gauss(4.2, 0.4), 2)),
    ("Chloride", "mmol/L", 98, 107, lambda: round(random.gauss(102, 3), 1)),
    ("CO2 (Bicarbonate)", "mmol/L", 23, 29, lambda: round(random.gauss(26, 2.5), 1)),
    ("BUN", "mg/dL", 7, 20, lambda: round(random.gauss(15, 6), 1)),
    ("Creatinine", "mg/dL", 0.6, 1.3, lambda: round(random.gauss(0.95, 0.3), 2)),
    ("eGFR", "mL/min/1.73m2", 90, 120, lambda: round(random.gauss(88, 20), 0)),
    ("ALT (SGPT)", "U/L", 7, 56, lambda: round(random.gauss(30, 18), 0)),
    ("AST (SGOT)", "U/L", 10, 40, lambda: round(random.gauss(26, 14), 0)),
    ("TSH", "mIU/L", 0.4, 4.0, lambda: round(random.gauss(2.3, 1.8), 2)),
    ("Free T4", "ng/dL", 0.8, 1.8, lambda: round(random.gauss(1.2, 0.3), 2)),
    ("WBC Count", "10^3/uL", 4.5, 11.0, lambda: round(random.gauss(7.2, 2.3), 1)),
    ("RBC Count", "10^6/uL", 4.2, 5.9, lambda: round(random.gauss(4.9, 0.5), 2)),
    ("Hemoglobin", "g/dL", 12.0, 17.5, lambda: round(random.gauss(14.2, 1.8), 1)),
    ("Hematocrit", "%", 36.0, 50.0, lambda: round(random.gauss(43, 5), 1)),
    ("Platelet Count", "10^3/uL", 150, 450, lambda: round(random.gauss(280, 80), 0)),
    ("Vitamin D, 25-OH", "ng/mL", 30, 100, lambda: round(random.gauss(34, 15), 1)),
    ("Ferritin", "ng/mL", 20, 250, lambda: round(random.gauss(90, 60), 0)),
    ("Vitamin B12", "pg/mL", 200, 900, lambda: round(random.gauss(480, 180), 0)),
    ("C-Reactive Protein", "mg/L", 0.0, 3.0, lambda: round(abs(random.gauss(1.8, 2.5)), 2)),
]

# ==================================================================
# 4. PATIENTS
# ==================================================================
INSURANCE_PROVIDERS = [
    "Aetna", "Blue Cross Blue Shield", "UnitedHealthcare", "Cigna", "Humana",
    "Kaiser Permanente", "Medicare", "Medicaid", "self-pay",
]

patients = []
for i in range(1, N_PATIENTS + 1):
    pid = f"PT-{i:05d}"
    sex = random.choice(["M", "F"])
    first = fake.first_name_male() if sex == "M" else fake.first_name_female()
    last = fake.last_name()
    dob_dt = fake.date_of_birth(minimum_age=0, maximum_age=95)
    phone = fake.numerify("(###) ###-####")
    email = None if random.random() < 0.08 else f"{first.lower()}.{last.lower()}{random.randint(1,999)}@{fake.free_email_domain()}"
    address = f"{fake.street_address()}, {fake.city()}, {fake.state_abbr()} {fake.zipcode()}"
    insurance = random.choice(INSURANCE_PROVIDERS)
    created = random_past_datetime(days_back_max=2600, days_back_min=1)
    updated = random_dt_between(created, now)
    patients.append((
        pid, first, last, iso_date(dob_dt), sex, phone, email, address,
        insurance, iso_dt(created), iso_dt(updated), dob_dt
    ))

# ==================================================================
# 5. APPOINTMENTS
# ==================================================================
APPT_REASONS = [
    "Annual physical exam", "Follow-up visit", "New patient consultation",
    "Hypertension management", "Diabetes follow-up", "Medication review",
    "Lab results review", "Cold/flu symptoms", "Sore throat", "Back pain",
    "Joint pain evaluation", "Skin rash", "Anxiety/depression check-in",
    "Well-child visit", "Vaccination", "Pre-operative clearance",
    "Post-operative follow-up", "Chest pain evaluation", "Headache workup",
    "Allergy consultation", "UTI symptoms", "Fatigue evaluation",
    "Thyroid follow-up", "Cholesterol management", "Prenatal checkup",
    "Asthma follow-up", "GERD/reflux symptoms", "Ear pain", "Sports physical",
    "Medication refill",
]

STATUS_WEIGHTS = [("completed", 0.62), ("scheduled", 0.20), ("cancelled", 0.10), ("no_show", 0.08)]
status_pool = [s for s, w in STATUS_WEIGHTS for _ in range(int(w * 100))]

appointments = []
appt_by_patient = {}
for i in range(1, N_APPOINTMENTS + 1):
    aid = f"AP-{i:06d}"
    patient = random.choice(patients)
    provider = random.choice(providers)
    status = random.choice(status_pool)
    if status in ("completed", "no_show", "cancelled"):
        sched = random_past_datetime(days_back_max=900, days_back_min=1)
    else:  # scheduled -> future
        sched = now + timedelta(days=random.randint(1, 90), hours=random.randint(0, 8))
    reason = random.choice(APPT_REASONS)
    created = sched - timedelta(days=random.randint(1, 30))
    if created > now:
        created = now - timedelta(days=random.randint(1, 5))
    updated = created if status == "scheduled" else random_dt_between(created, min(sched, now) if sched < now else now)
    appointments.append((aid, patient[0], provider[0], iso_dt(sched), reason, status, iso_dt(created), iso_dt(updated)))
    appt_by_patient.setdefault(patient[0], []).append((sched, status))

# ==================================================================
# 6. PRESCRIPTIONS
# ==================================================================
prescriptions = []
for i in range(1, N_PRESCRIPTIONS + 1):
    rxid = f"RX-{i:06d}"
    patient = random.choice(patients)
    provider = random.choice(providers)
    med = random.choice(medications_reference)
    med_name, drug_class, typical_dosage, notes = med
    dosage = typical_dosage  # keep clinically consistent with reference

    start = random_past_datetime(days_back_max=1500, days_back_min=1)
    status_roll = random.random()
    if status_roll < 0.55:
        status = "active"
        end_date = None
    elif status_roll < 0.80:
        status = "completed"
        end_dt = start + timedelta(days=random.randint(7, 180))
        end_date = iso_date(min(end_dt, now))
    else:
        status = "discontinued"
        end_dt = start + timedelta(days=random.randint(3, 120))
        end_date = iso_date(min(end_dt, now))

    created = start
    updated = created if end_date is None else random_dt_between(created, now)
    prescriptions.append((
        rxid, patient[0], provider[0], med_name, dosage, iso_date(start),
        end_date, status, iso_dt(created), iso_dt(updated)
    ))

# ==================================================================
# 7. LAB RESULTS
# ==================================================================
lab_results = []
for i in range(1, N_LAB_RESULTS + 1):
    lid = f"LB-{i:06d}"
    patient = random.choice(patients)
    test_name, unit, ref_low, ref_high, gen = random.choice(LAB_TESTS)
    value = gen()
    if value < 0:
        value = abs(value)
    flagged = 1 if (value < ref_low or value > ref_high) else 0
    recorded = random_past_datetime(days_back_max=1200, days_back_min=0)
    lab_results.append((lid, patient[0], test_name, value, unit, ref_low, ref_high, flagged, iso_dt(recorded)))

# ==================================================================
# 8. AGENT MEMORY
# ==================================================================
MEMORY_TEMPLATES = {
    "preference": [
        "Prefers morning appointments before 10am.",
        "Prefers text message reminders over phone calls.",
        "Requests a female provider when possible.",
        "Prefers telehealth visits over in-person when available.",
        "Likes detailed explanations of lab results in plain language.",
        "Prefers appointments scheduled on Fridays.",
        "Wants a family member included in appointment communications.",
        "Prefers generic medications over brand-name when available.",
        "Requests appointment reminders 2 days in advance.",
        "Prefers the same provider for continuity of care.",
    ],
    "recurring_topic": [
        "Frequently asks about medication side effects before starting new prescriptions.",
        "Recurring concerns about blood pressure readings at home.",
        "Regularly brings up sleep quality during visits.",
        "Often discusses stress and work-life balance.",
        "Repeatedly asks about interactions between current medications.",
        "Recurring questions about diet and weight management.",
        "Frequently mentions joint stiffness in the mornings.",
        "Ongoing discussion about smoking cessation progress.",
        "Regularly asks about vaccine schedules for children.",
        "Recurring follow-up on exercise tolerance post-cardiac event.",
    ],
    "flag": [
        "Documented penicillin allergy - avoid beta-lactam antibiotics.",
        "History of medication non-adherence - consider simplified regimen.",
        "Flagged for missed appointments - two no-shows in past 6 months.",
        "Sulfa drug allergy noted in intake form.",
        "Latex allergy - use non-latex materials during visits.",
        "Flagged for elevated fall risk during last assessment.",
        "History of adverse reaction to NSAIDs (GI bleeding).",
        "Flagged for overdue annual labs - last panel over 14 months ago.",
        "Contrast dye allergy noted for imaging studies.",
        "Flagged for language interpreter needed at visits.",
    ],
}

agent_memory = []
for i in range(1, N_MEMORIES + 1):
    mid = f"MEM-{i:06d}"
    patient = random.choice(patients)
    mtype = random.choices(
        ["preference", "recurring_topic", "flag"], weights=[0.4, 0.35, 0.25]
    )[0]
    content = random.choice(MEMORY_TEMPLATES[mtype])
    created = random_past_datetime(days_back_max=800, days_back_min=0)
    agent_memory.append((mid, patient[0], mtype, content, iso_dt(created)))

# ==================================================================
# BUILD DATABASE
# ==================================================================
with open(SCHEMA_PATH, "r") as f:
    schema_sql = f.read()

conn = sqlite3.connect(DB_PATH)
conn.executescript(schema_sql)
cur = conn.cursor()

cur.executemany(
    "INSERT INTO providers (provider_id, name, specialty, created_at) VALUES (?, ?, ?, ?)",
    providers,
)

cur.executemany(
    """INSERT INTO patients
       (patient_id, first_name, last_name, dob, sex, phone, email, address,
        insurance_provider, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    [p[:11] for p in patients],
)

cur.executemany(
    """INSERT INTO medications_reference
       (medication_name, drug_class, typical_dosage, notes)
       VALUES (?, ?, ?, ?)""",
    medications_reference,
)

cur.executemany(
    """INSERT INTO appointments
       (appointment_id, patient_id, provider_id, scheduled_time, reason,
        status, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
    appointments,
)

cur.executemany(
    """INSERT INTO prescriptions
       (prescription_id, patient_id, provider_id, medication_name, dosage,
        start_date, end_date, status, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    prescriptions,
)

cur.executemany(
    """INSERT INTO lab_results
       (lab_result_id, patient_id, test_name, value, unit, reference_low,
        reference_high, flagged, recorded_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    lab_results,
)

cur.executemany(
    """INSERT INTO agent_memory
       (memory_id, patient_id, memory_type, content, created_at)
       VALUES (?, ?, ?, ?, ?)""",
    agent_memory,
)

conn.commit()

# quick sanity report
for tbl in ["patients", "providers", "appointments", "medications_reference",
            "prescriptions", "lab_results", "agent_memory"]:
    n = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"{tbl:25s} {n:6d} rows")

conn.close()
print(f"\nDatabase written to {DB_PATH}")