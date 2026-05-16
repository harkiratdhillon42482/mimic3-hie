-- =============================================================================
-- hie_schema.sql
-- OMOP CDM-aligned analytics layer for MIMIC-III HIE
-- Lives in MIMICold database, hie schema
-- Source of truth: YottaDB ^PHD globals
-- This schema is a DERIVED VIEW — never write to it directly
-- Sync from ^PHD via sync_engine.py
--
-- Run as postgres user:
--   psql -U postgres -d MIMICold -f hie_schema.sql
-- =============================================================================

-- Create schema
CREATE SCHEMA IF NOT EXISTS hie;

-- Set search path for this session
SET search_path TO hie, public;

-- =============================================================================
-- SOURCE REGISTRY
-- Tracks which datasets have been loaded
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.source (
    source_id       SERIAL PRIMARY KEY,
    source_code     VARCHAR(10)  UNIQUE NOT NULL,  -- MIII, MIV, SYN
    source_name     VARCHAR(100) NOT NULL,
    source_version  VARCHAR(20),
    loaded_at       TIMESTAMPTZ  DEFAULT NOW(),
    record_count    BIGINT       DEFAULT 0,
    notes           TEXT
);

INSERT INTO hie.source (source_code, source_name, source_version, notes)
VALUES ('MIII', 'MIMIC-III', '1.4', 'PhysioNet MIMIC-III Clinical Database')
ON CONFLICT (source_code) DO NOTHING;

-- =============================================================================
-- PERSON  (OMOP: person)
-- One row per patient — canonical identity
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.person (
    person_id               VARCHAR(20)  PRIMARY KEY,  -- HIE-MIII-XXXXXX
    source_id               INT          NOT NULL REFERENCES hie.source(source_id),
    -- Source identifiers
    src_subject_id          VARCHAR(20)  NOT NULL,
    -- OMOP standard fields
    gender_concept_code     VARCHAR(10),               -- M/F/U
    year_of_birth           INT,
    month_of_birth          INT,
    day_of_birth            INT,
    birth_datetime          TIMESTAMPTZ,
    death_datetime          TIMESTAMPTZ,
    -- MIMIC-specific
    dob_raw                 VARCHAR(20),               -- raw MIMIC DOB (shifted)
    dod_raw                 VARCHAR(20),
    dod_hosp                DATE,
    dod_ssn                 DATE,
    deceased                BOOLEAN      DEFAULT FALSE,
    expire_flag             SMALLINT,
    -- Audit
    ydb_pid                 VARCHAR(20),               -- ^PHD key
    synced_at               TIMESTAMPTZ  DEFAULT NOW(),
    created_at              TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_person_src    ON hie.person(src_subject_id);
CREATE INDEX IF NOT EXISTS idx_person_source ON hie.person(source_id);
CREATE INDEX IF NOT EXISTS idx_person_dead   ON hie.person(deceased);
CREATE INDEX IF NOT EXISTS idx_person_gender ON hie.person(gender_concept_code);

-- =============================================================================
-- VISIT OCCURRENCE  (OMOP: visit_occurrence)
-- One row per hospital admission
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.visit_occurrence (
    visit_occurrence_id     VARCHAR(20)  PRIMARY KEY,  -- ENC-MIII-XXXXXX
    person_id               VARCHAR(20)  NOT NULL REFERENCES hie.person(person_id),
    source_id               INT          NOT NULL REFERENCES hie.source(source_id),
    -- Source identifiers
    src_hadm_id             VARCHAR(20)  NOT NULL,
    -- OMOP standard fields
    visit_concept_code      VARCHAR(50),               -- INPATIENT/EMERGENCY/etc
    visit_start_datetime    TIMESTAMPTZ,
    visit_end_datetime      TIMESTAMPTZ,
    visit_type              VARCHAR(50),               -- admission_type
    -- Clinical details
    admit_source            VARCHAR(100),
    discharge_disposition   VARCHAR(100),
    insurance               VARCHAR(50),
    language                VARCHAR(20),
    religion                VARCHAR(50),
    marital_status          VARCHAR(20),
    ethnicity               VARCHAR(100),
    admit_diagnosis_text    TEXT,
    -- Derived metrics
    los_hours               NUMERIC(8,2),
    los_days                NUMERIC(6,2),
    hospital_expire_flag    SMALLINT,
    has_icu_stay            BOOLEAN      DEFAULT FALSE,
    -- Audit
    ydb_eid                 VARCHAR(20),
    synced_at               TIMESTAMPTZ  DEFAULT NOW(),
    created_at              TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vo_person   ON hie.visit_occurrence(person_id);
CREATE INDEX IF NOT EXISTS idx_vo_hadm     ON hie.visit_occurrence(src_hadm_id);
CREATE INDEX IF NOT EXISTS idx_vo_start    ON hie.visit_occurrence(visit_start_datetime);
CREATE INDEX IF NOT EXISTS idx_vo_type     ON hie.visit_occurrence(visit_type);
CREATE INDEX IF NOT EXISTS idx_vo_expire   ON hie.visit_occurrence(hospital_expire_flag);
CREATE INDEX IF NOT EXISTS idx_vo_icu      ON hie.visit_occurrence(has_icu_stay);

-- =============================================================================
-- VISIT DETAIL  (OMOP: visit_detail)
-- ICU stays — sub-visits within an admission
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.visit_detail (
    visit_detail_id         SERIAL       PRIMARY KEY,
    visit_occurrence_id     VARCHAR(20)  NOT NULL REFERENCES hie.visit_occurrence(visit_occurrence_id),
    person_id               VARCHAR(20)  NOT NULL REFERENCES hie.person(person_id),
    source_id               INT          NOT NULL REFERENCES hie.source(source_id),
    -- Source
    src_icustay_id          VARCHAR(20),
    -- ICU details
    care_unit               VARCHAR(50),               -- MICU/SICU/CCU/NICU/TSICU
    first_care_unit         VARCHAR(50),
    last_care_unit          VARCHAR(50),
    intime                  TIMESTAMPTZ,
    outtime                 TIMESTAMPTZ,
    los_hours               NUMERIC(8,2),
    db_source               VARCHAR(20),               -- carevue/metavision
    synced_at               TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vd_visit  ON hie.visit_detail(visit_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_vd_person ON hie.visit_detail(person_id);
CREATE INDEX IF NOT EXISTS idx_vd_unit   ON hie.visit_detail(care_unit);

-- =============================================================================
-- CONDITION OCCURRENCE  (OMOP: condition_occurrence)
-- Diagnoses — ICD-9 coded
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.condition_occurrence (
    condition_occurrence_id BIGSERIAL    PRIMARY KEY,
    visit_occurrence_id     VARCHAR(20)  NOT NULL REFERENCES hie.visit_occurrence(visit_occurrence_id),
    person_id               VARCHAR(20)  NOT NULL REFERENCES hie.person(person_id),
    source_id               INT          NOT NULL REFERENCES hie.source(source_id),
    -- OMOP standard
    condition_concept_code  VARCHAR(20),               -- ICD9 code
    condition_coding_system VARCHAR(20)  DEFAULT 'ICD9CM',
    condition_start_date    DATE,
    condition_type          VARCHAR(20),               -- primary/secondary
    -- MIMIC-specific
    seq_num                 SMALLINT,                  -- 1 = primary diagnosis
    short_title             VARCHAR(100),
    long_title              VARCHAR(300),
    synced_at               TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_co_visit  ON hie.condition_occurrence(visit_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_co_person ON hie.condition_occurrence(person_id);
CREATE INDEX IF NOT EXISTS idx_co_icd9   ON hie.condition_occurrence(condition_concept_code);
CREATE INDEX IF NOT EXISTS idx_co_seq    ON hie.condition_occurrence(seq_num);
-- Partial index for primary diagnoses only
CREATE INDEX IF NOT EXISTS idx_co_primary ON hie.condition_occurrence(condition_concept_code)
    WHERE seq_num = 1;

-- =============================================================================
-- PROCEDURE OCCURRENCE  (OMOP: procedure_occurrence)
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.procedure_occurrence (
    procedure_occurrence_id BIGSERIAL    PRIMARY KEY,
    visit_occurrence_id     VARCHAR(20)  NOT NULL REFERENCES hie.visit_occurrence(visit_occurrence_id),
    person_id               VARCHAR(20)  NOT NULL REFERENCES hie.person(person_id),
    source_id               INT          NOT NULL REFERENCES hie.source(source_id),
    procedure_concept_code  VARCHAR(20),
    procedure_coding_system VARCHAR(20)  DEFAULT 'ICD9CM',
    procedure_date          DATE,
    seq_num                 SMALLINT,
    short_title             VARCHAR(100),
    long_title              VARCHAR(300),
    synced_at               TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_po_visit  ON hie.procedure_occurrence(visit_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_po_person ON hie.procedure_occurrence(person_id);
CREATE INDEX IF NOT EXISTS idx_po_code   ON hie.procedure_occurrence(procedure_concept_code);

-- =============================================================================
-- DRUG EXPOSURE  (OMOP: drug_exposure)
-- Prescriptions / medications
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.drug_exposure (
    drug_exposure_id        BIGSERIAL    PRIMARY KEY,
    visit_occurrence_id     VARCHAR(20)  NOT NULL REFERENCES hie.visit_occurrence(visit_occurrence_id),
    person_id               VARCHAR(20)  NOT NULL REFERENCES hie.person(person_id),
    source_id               INT          NOT NULL REFERENCES hie.source(source_id),
    -- Drug identification
    drug_name               VARCHAR(200),
    drug_name_generic       VARCHAR(200),
    drug_type               VARCHAR(30),               -- MAIN/BASE/ADDITIVE
    ndc                     VARCHAR(20),
    formulary_drug_cd       VARCHAR(20),
    prod_strength           VARCHAR(100),
    -- Dosing
    dose_val                VARCHAR(50),
    dose_unit               VARCHAR(30),
    route                   VARCHAR(30),
    -- Timing
    drug_exposure_start     TIMESTAMPTZ,
    drug_exposure_end       TIMESTAMPTZ,
    synced_at               TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_de_visit  ON hie.drug_exposure(visit_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_de_person ON hie.drug_exposure(person_id);
CREATE INDEX IF NOT EXISTS idx_de_drug   ON hie.drug_exposure(drug_name);
CREATE INDEX IF NOT EXISTS idx_de_ndc    ON hie.drug_exposure(ndc) WHERE ndc IS NOT NULL;

-- =============================================================================
-- MEASUREMENT  (OMOP: measurement)
-- Lab results
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.measurement (
    measurement_id          BIGSERIAL    PRIMARY KEY,
    visit_occurrence_id     VARCHAR(20)  REFERENCES hie.visit_occurrence(visit_occurrence_id),
    person_id               VARCHAR(20)  NOT NULL REFERENCES hie.person(person_id),
    source_id               INT          NOT NULL REFERENCES hie.source(source_id),
    -- Lab identification
    src_itemid              INT,                       -- MIMIC itemid
    measurement_concept     VARCHAR(100),              -- lab label
    loinc_code              VARCHAR(20),               -- where mapped
    -- Values
    measurement_datetime    TIMESTAMPTZ,
    value_as_number         NUMERIC(18,6),
    value_as_string         VARCHAR(200),
    unit_concept            VARCHAR(30),
    range_low               NUMERIC(18,6),
    range_high              NUMERIC(18,6),
    -- Flags
    abnormal_flag           VARCHAR(20),               -- H/L/HH/LL/abnormal
    -- Classification
    fluid                   VARCHAR(50),               -- blood/urine/CSF
    category                VARCHAR(50),
    synced_at               TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_meas_visit  ON hie.measurement(visit_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_meas_person ON hie.measurement(person_id);
CREATE INDEX IF NOT EXISTS idx_meas_item   ON hie.measurement(src_itemid);
CREATE INDEX IF NOT EXISTS idx_meas_dt     ON hie.measurement(measurement_datetime);
CREATE INDEX IF NOT EXISTS idx_meas_flag   ON hie.measurement(abnormal_flag)
    WHERE abnormal_flag IS NOT NULL;

-- =============================================================================
-- NOTE  (OMOP: note)
-- Clinical notes — discharge summaries, radiology, nursing etc
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.note (
    note_id                 BIGSERIAL    PRIMARY KEY,
    visit_occurrence_id     VARCHAR(20)  REFERENCES hie.visit_occurrence(visit_occurrence_id),
    person_id               VARCHAR(20)  NOT NULL REFERENCES hie.person(person_id),
    source_id               INT          NOT NULL REFERENCES hie.source(source_id),
    -- Note metadata
    note_category           VARCHAR(50),               -- Discharge summary/Radiology/etc
    note_description        VARCHAR(100),
    note_datetime           TIMESTAMPTZ,
    cgid                    INT,
    -- Content
    note_text               TEXT,
    char_count              INT,
    token_estimate          INT,                       -- char_count / 4
    -- Quality
    is_error                BOOLEAN      DEFAULT FALSE,
    -- Index for FTS
    note_tsv                TSVECTOR,
    synced_at               TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_note_visit    ON hie.note(visit_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_note_person   ON hie.note(person_id);
CREATE INDEX IF NOT EXISTS idx_note_category ON hie.note(note_category);
CREATE INDEX IF NOT EXISTS idx_note_dt       ON hie.note(note_datetime);
-- Full text search index
CREATE INDEX IF NOT EXISTS idx_note_fts      ON hie.note
    USING GIN(note_tsv);

-- Auto-update TSV on insert/update
CREATE OR REPLACE FUNCTION hie.note_tsv_update() RETURNS TRIGGER AS $$
BEGIN
    NEW.note_tsv := to_tsvector('english',
        COALESCE(NEW.note_category, '') || ' ' ||
        COALESCE(NEW.note_description, '') || ' ' ||
        COALESCE(NEW.note_text, ''));
    NEW.char_count := length(COALESCE(NEW.note_text, ''));
    NEW.token_estimate := length(COALESCE(NEW.note_text, '')) / 4;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER note_tsv_trigger
    BEFORE INSERT OR UPDATE ON hie.note
    FOR EACH ROW EXECUTE FUNCTION hie.note_tsv_update();

-- =============================================================================
-- COHORT  (LLM training cohort management)
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.cohort (
    cohort_id               SERIAL       PRIMARY KEY,
    cohort_name             VARCHAR(100) UNIQUE NOT NULL,
    description             TEXT,
    sql_definition          TEXT,                      -- the query used
    source_codes            VARCHAR(50)[],             -- e.g. {MIII, MIV}
    created_at              TIMESTAMPTZ  DEFAULT NOW(),
    person_count            INT          DEFAULT 0,
    notes                   TEXT
);

CREATE TABLE IF NOT EXISTS hie.cohort_member (
    cohort_member_id        BIGSERIAL    PRIMARY KEY,
    cohort_id               INT          NOT NULL REFERENCES hie.cohort(cohort_id),
    person_id               VARCHAR(20)  NOT NULL REFERENCES hie.person(person_id),
    split                   VARCHAR(10)  NOT NULL DEFAULT 'train', -- train/val/test
    oversample_weight       NUMERIC(5,2) DEFAULT 1.0,
    added_at                TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (cohort_id, person_id)
);

CREATE INDEX IF NOT EXISTS idx_cm_cohort ON hie.cohort_member(cohort_id);
CREATE INDEX IF NOT EXISTS idx_cm_split  ON hie.cohort_member(cohort_id, split);
CREATE INDEX IF NOT EXISTS idx_cm_person ON hie.cohort_member(person_id);

-- =============================================================================
-- TRAINING RUN  (experiment tracking)
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.training_run (
    run_id                  VARCHAR(50)  PRIMARY KEY,
    cohort_id               INT          REFERENCES hie.cohort(cohort_id),
    model_name              VARCHAR(100),
    epoch                   INT,
    train_loss              NUMERIC(10,6),
    val_loss                NUMERIC(10,6),
    perplexity              NUMERIC(10,4),
    n_patients              INT,
    n_tokens                BIGINT,
    config_json             JSONB,
    started_at              TIMESTAMPTZ  DEFAULT NOW(),
    completed_at            TIMESTAMPTZ,
    notes                   TEXT
);

-- =============================================================================
-- SYNC STATE  (tracks ^PHD → Postgres sync progress)
-- =============================================================================

CREATE TABLE IF NOT EXISTS hie.sync_state (
    source_code             VARCHAR(10)  PRIMARY KEY,
    last_sync               TIMESTAMPTZ  DEFAULT '1970-01-01',
    last_pid                VARCHAR(20)  DEFAULT '',
    records_synced          BIGINT       DEFAULT 0,
    status                  VARCHAR(20)  DEFAULT 'idle',
    last_error              TEXT,
    updated_at              TIMESTAMPTZ  DEFAULT NOW()
);

INSERT INTO hie.sync_state (source_code) VALUES ('MIII')
ON CONFLICT (source_code) DO NOTHING;

-- =============================================================================
-- USEFUL ANALYTICS VIEWS
-- =============================================================================

-- Patient summary — one row per patient with all counts
CREATE OR REPLACE VIEW hie.v_patient_summary AS
SELECT
    p.person_id,
    p.src_subject_id,
    p.gender_concept_code                   AS sex,
    p.birth_datetime,
    p.death_datetime,
    p.deceased,
    COUNT(DISTINCT vo.visit_occurrence_id)  AS n_admissions,
    COUNT(DISTINCT vd.visit_detail_id)      AS n_icu_stays,
    MIN(vo.visit_start_datetime)            AS first_admit,
    MAX(vo.visit_end_datetime)              AS last_discharge,
    ROUND(SUM(vo.los_hours)::numeric, 1)    AS total_los_hours,
    COUNT(DISTINCT co.condition_concept_code) AS n_unique_dx,
    COUNT(DISTINCT de.drug_name)            AS n_unique_drugs,
    COUNT(DISTINCT m.measurement_id)        AS n_lab_results,
    COUNT(DISTINCT n.note_id)               AS n_notes,
    COUNT(DISTINCT CASE WHEN n.note_category = 'Discharge summary'
          THEN n.note_id END)               AS n_discharge_summaries
FROM hie.person p
LEFT JOIN hie.visit_occurrence vo   ON p.person_id = vo.person_id
LEFT JOIN hie.visit_detail vd       ON vo.visit_occurrence_id = vd.visit_occurrence_id
LEFT JOIN hie.condition_occurrence co ON vo.visit_occurrence_id = co.visit_occurrence_id
LEFT JOIN hie.drug_exposure de      ON vo.visit_occurrence_id = de.visit_occurrence_id
LEFT JOIN hie.measurement m         ON vo.visit_occurrence_id = m.visit_occurrence_id
LEFT JOIN hie.note n                ON p.person_id = n.person_id
GROUP BY p.person_id, p.src_subject_id, p.gender_concept_code,
         p.birth_datetime, p.death_datetime, p.deceased;

-- Encounter summary — rich view for cohort building
CREATE OR REPLACE VIEW hie.v_encounter_summary AS
SELECT
    vo.visit_occurrence_id,
    vo.person_id,
    p.gender_concept_code                   AS sex,
    vo.visit_start_datetime                 AS admit_dt,
    vo.visit_end_datetime                   AS discharge_dt,
    vo.los_hours,
    ROUND((vo.los_hours / 24)::numeric, 1)  AS los_days,
    vo.visit_type                           AS admission_type,
    vo.insurance,
    vo.ethnicity,
    vo.hospital_expire_flag,
    vo.has_icu_stay,
    -- Primary diagnosis
    co_pri.condition_concept_code           AS primary_icd9,
    co_pri.short_title                      AS primary_dx,
    -- Counts
    COUNT(DISTINCT co.condition_occurrence_id)  AS n_diagnoses,
    COUNT(DISTINCT de.drug_exposure_id)         AS n_medications,
    COUNT(DISTINCT m.measurement_id)            AS n_labs,
    COUNT(DISTINCT n.note_id)                   AS n_notes,
    COUNT(DISTINCT vd.visit_detail_id)          AS n_icu_stays
FROM hie.visit_occurrence vo
JOIN hie.person p               ON vo.person_id = p.person_id
LEFT JOIN hie.condition_occurrence co
    ON vo.visit_occurrence_id = co.visit_occurrence_id
LEFT JOIN hie.condition_occurrence co_pri
    ON vo.visit_occurrence_id = co_pri.visit_occurrence_id
    AND co_pri.seq_num = 1
LEFT JOIN hie.drug_exposure de  ON vo.visit_occurrence_id = de.visit_occurrence_id
LEFT JOIN hie.measurement m     ON vo.visit_occurrence_id = m.visit_occurrence_id
LEFT JOIN hie.note n            ON vo.visit_occurrence_id = n.visit_occurrence_id
LEFT JOIN hie.visit_detail vd   ON vo.visit_occurrence_id = vd.visit_occurrence_id
GROUP BY vo.visit_occurrence_id, vo.person_id, p.gender_concept_code,
         vo.visit_start_datetime, vo.visit_end_datetime, vo.los_hours,
         vo.visit_type, vo.insurance, vo.ethnicity,
         vo.hospital_expire_flag, vo.has_icu_stay,
         co_pri.condition_concept_code, co_pri.short_title;

-- Discharge summaries only — most useful for LLM training
CREATE OR REPLACE VIEW hie.v_discharge_summary AS
SELECT
    n.note_id,
    n.person_id,
    n.visit_occurrence_id,
    n.note_datetime,
    n.note_text,
    n.char_count,
    n.token_estimate,
    vo.visit_type,
    vo.los_hours,
    vo.hospital_expire_flag,
    co_pri.condition_concept_code   AS primary_icd9,
    co_pri.short_title              AS primary_dx
FROM hie.note n
JOIN hie.visit_occurrence vo
    ON n.visit_occurrence_id = vo.visit_occurrence_id
LEFT JOIN hie.condition_occurrence co_pri
    ON vo.visit_occurrence_id = co_pri.visit_occurrence_id
    AND co_pri.seq_num = 1
WHERE n.note_category = 'Discharge summary'
  AND n.is_error = FALSE;

-- =============================================================================
-- GRANT ACCESS
-- =============================================================================

GRANT USAGE ON SCHEMA hie TO postgres;
GRANT ALL ON ALL TABLES IN SCHEMA hie TO postgres;
GRANT ALL ON ALL SEQUENCES IN SCHEMA hie TO postgres;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA hie TO postgres;

-- =============================================================================
-- SUMMARY
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '===========================================';
    RAISE NOTICE '  HIE schema created successfully';
    RAISE NOTICE '  Tables: person, visit_occurrence,';
    RAISE NOTICE '          visit_detail, condition_occurrence,';
    RAISE NOTICE '          procedure_occurrence, drug_exposure,';
    RAISE NOTICE '          measurement, note, cohort,';
    RAISE NOTICE '          cohort_member, training_run, sync_state';
    RAISE NOTICE '  Views:  v_patient_summary,';
    RAISE NOTICE '          v_encounter_summary,';
    RAISE NOTICE '          v_discharge_summary';
    RAISE NOTICE '===========================================';
END $$;
