"""SQL-based feature store builder lifted directly from the original notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import duckdb
import pandas as pd

from creditrisk.features.preprocess import EPS_DEFAULT

APPLICATION_SQL = """
SELECT

    ----------------- as-is features ---------------

    SK_ID_CURR,
    TARGET,

    EXT_SOURCE_1,
    EXT_SOURCE_2,
    EXT_SOURCE_3,

    AMT_ANNUITY,
    AMT_CREDIT,
    AMT_GOODS_PRICE,
    AMT_INCOME_TOTAL,

    DAYS_BIRTH,
    DAYS_EMPLOYED,
    DAYS_REGISTRATION,
    DAYS_ID_PUBLISH,
    DAYS_LAST_PHONE_CHANGE,

    CODE_GENDER,

    CNT_CHILDREN,
    CNT_FAM_MEMBERS,

    REGION_RATING_CLIENT,
    REGION_RATING_CLIENT_W_CITY,
    REGION_POPULATION_RELATIVE,

    WEEKDAY_APPR_PROCESS_START,
    HOUR_APPR_PROCESS_START,

    NAME_INCOME_TYPE,
    NAME_EDUCATION_TYPE,
    NAME_HOUSING_TYPE,
    NAME_CONTRACT_TYPE,
    NAME_FAMILY_STATUS,



    REG_REGION_NOT_LIVE_REGION,
    REG_REGION_NOT_WORK_REGION,
    LIVE_REGION_NOT_WORK_REGION,
    REG_CITY_NOT_LIVE_CITY,
    REG_CITY_NOT_WORK_CITY,
    LIVE_CITY_NOT_WORK_CITY,

    OBS_30_CNT_SOCIAL_CIRCLE,
    OBS_60_CNT_SOCIAL_CIRCLE,
    DEF_30_CNT_SOCIAL_CIRCLE,
    DEF_60_CNT_SOCIAL_CIRCLE,

    AMT_REQ_CREDIT_BUREAU_HOUR,
    AMT_REQ_CREDIT_BUREAU_DAY,
    AMT_REQ_CREDIT_BUREAU_WEEK,
    AMT_REQ_CREDIT_BUREAU_MON,
    AMT_REQ_CREDIT_BUREAU_QRT,
    AMT_REQ_CREDIT_BUREAU_YEAR,

    FLAG_MOBIL,
    FLAG_EMP_PHONE,
    FLAG_WORK_PHONE,
    FLAG_CONT_MOBILE,
    FLAG_PHONE,
    FLAG_EMAIL,

    FLAG_OWN_CAR,
    FLAG_OWN_REALTY,

    FLAG_DOCUMENT_2,
    FLAG_DOCUMENT_3,
    FLAG_DOCUMENT_4,
    FLAG_DOCUMENT_5,
    FLAG_DOCUMENT_6,
    FLAG_DOCUMENT_7,
    FLAG_DOCUMENT_8,
    FLAG_DOCUMENT_9,
    FLAG_DOCUMENT_10,
    FLAG_DOCUMENT_11,
    FLAG_DOCUMENT_12,
    FLAG_DOCUMENT_13,
    FLAG_DOCUMENT_14,
    FLAG_DOCUMENT_15,
    FLAG_DOCUMENT_16,
    FLAG_DOCUMENT_17,
    FLAG_DOCUMENT_18,
    FLAG_DOCUMENT_19,
    FLAG_DOCUMENT_20,
    FLAG_DOCUMENT_21,

    TOT_MISSING_COUNT,

    ----------------- modified time features ---------------

    -DAYS_BIRTH/(365 + {EPS}) AS AGE_YEARS,
    CASE WHEN DAYS_EMPLOYED = 365243 THEN 1 ELSE 0 END AS DAYS_EMPLOYED_ANOMALY,
    -CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END / (365 + {EPS}) AS EMPLOYED_YEARS,
    -CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END / (-DAYS_BIRTH + {EPS}) AS EMPLOYMENT_YEARS_TO_AGE,

    ----------------- ratio features ---------------

    AMT_ANNUITY/(AMT_CREDIT + {EPS}) AS PAYMENT_RATE,
    AMT_CREDIT/(AMT_INCOME_TOTAL + {EPS}) AS CREDIT_TO_INCOME,
    AMT_ANNUITY/(AMT_INCOME_TOTAL + {EPS}) AS ANNUITY_TO_INCOME,
    AMT_GOODS_PRICE/(AMT_CREDIT + {EPS}) AS GOODS_TO_CREDIT,
    AMT_INCOME_TOTAL/(CNT_FAM_MEMBERS + {EPS}) AS INCOME_PER_PERSON,
    CNT_CHILDREN/(CNT_FAM_MEMBERS + {EPS}) AS CHILDREN_RATIO,

    ----------------- missing-count features ---------------

    CASE WHEN EXT_SOURCE_1 IS NULL THEN 1 ELSE 0 END AS EXT_SOURCE_1_IS_MISSING,
    CASE WHEN EXT_SOURCE_2 IS NULL THEN 1 ELSE 0 END AS EXT_SOURCE_2_IS_MISSING,
    CASE WHEN EXT_SOURCE_3 IS NULL THEN 1 ELSE 0 END AS EXT_SOURCE_3_IS_MISSING,
    CASE WHEN OWN_CAR_AGE IS NULL THEN 1 ELSE 0 END AS  OWN_CAR_AGE_IS_MISSING,

    ----------------- count features ---------------

    COALESCE(FLAG_DOCUMENT_2,0) + COALESCE(FLAG_DOCUMENT_3,0) + COALESCE(FLAG_DOCUMENT_4,0) + COALESCE(FLAG_DOCUMENT_5,0) +
    COALESCE(FLAG_DOCUMENT_6,0) + COALESCE(FLAG_DOCUMENT_7,0) + COALESCE(FLAG_DOCUMENT_8,0) + COALESCE(FLAG_DOCUMENT_9,0) +
    COALESCE(FLAG_DOCUMENT_10,0) + COALESCE(FLAG_DOCUMENT_11,0) + COALESCE(FLAG_DOCUMENT_12,0) + COALESCE(FLAG_DOCUMENT_13,0) +
    COALESCE(FLAG_DOCUMENT_14,0) + COALESCE(FLAG_DOCUMENT_15,0) + COALESCE(FLAG_DOCUMENT_16,0) + COALESCE(FLAG_DOCUMENT_17,0) +
    COALESCE(FLAG_DOCUMENT_18,0) + COALESCE(FLAG_DOCUMENT_19,0) + COALESCE(FLAG_DOCUMENT_20,0) + COALESCE(FLAG_DOCUMENT_21,0)  AS DOC_COUNT,

    COALESCE(FLAG_MOBIL,0) + COALESCE(FLAG_EMP_PHONE,0) + COALESCE(FLAG_WORK_PHONE,0) +
    COALESCE(FLAG_CONT_MOBILE,0) + COALESCE(FLAG_PHONE,0) + COALESCE(FLAG_EMAIL,0) AS CONTACT_COUNT,

    COALESCE(REG_REGION_NOT_LIVE_REGION,0) + COALESCE(REG_REGION_NOT_WORK_REGION,0) + COALESCE(LIVE_REGION_NOT_WORK_REGION,0) +
    COALESCE(REG_CITY_NOT_LIVE_CITY,0) + COALESCE(REG_CITY_NOT_WORK_CITY,0) + COALESCE(LIVE_CITY_NOT_WORK_CITY,0) AS ADDR_MISMATCH_SUM

FROM
      application_df
"""

BUREAU_SQL = """
SELECT
      SK_ID_CURR,
      SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS BUREAU_N_ACTIVE,
      SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END) AS BUREAU_N_CLOSED,
      MAX(DAYS_CREDIT) AS BUREAU_LAST_CREDIT_DAYS,
      MAX(DAYS_CREDIT_UPDATE) AS BUREAU_LAST_UPDATE_DAYS,
      SUM(AMT_CREDIT_SUM_DEBT) AS BUREAU_TOTAL_DEBT,
      SUM(AMT_CREDIT_SUM) AS BUREAU_TOTAL_CREDIT,
      SUM(AMT_CREDIT_SUM_LIMIT) AS BUREAU_LIMIT_SUM,
      AVG(AMT_CREDIT_SUM_DEBT/(AMT_CREDIT_SUM + {EPS})) AS BUREAU_UTIL_MEAN,
      SUM(AMT_CREDIT_SUM_OVERDUE) AS BUREAU_OVERDUE_SUM,
      MAX(AMT_CREDIT_MAX_OVERDUE) AS BUREAU_MAX_OVERDUE,
      SUM(CNT_CREDIT_PROLONG) AS BUREAU_PROLONG_SUM
FROM
      bureau_df
GROUP BY
      SK_ID_CURR
"""

BUREAU_BALANCE_SQL = """
WITH table_1 AS (
    SELECT
          *,
          CASE WHEN (STATUS = 'C' OR STATUS = 'X' OR STATUS = '0') THEN 0 ELSE CAST(STATUS AS INT) END AS STATUS_NUM
    FROM
          bureau_balance_df
),

table_2 AS (
    SELECT
        *,
        CASE WHEN STATUS_NUM > 0 THEN 1 ELSE 0 END AS IS_DELINQ
    FROM
        table_1

),

table_3 AS (
    SELECT
        SK_ID_BUREAU,
        AVG(IS_DELINQ) AS BB_DELINQ_SHARE,
        MAX(STATUS_NUM) AS BB_WORST_STATUS_NUM
    FROM
        table_2
    GROUP BY
        SK_ID_BUREAU

),

table_4 AS (
    SELECT
          SK_ID_BUREAU,
          -MAX(MONTHS_BALANCE) AS BB_MONTHS_LAST_DELINQ
    FROM
          table_2
    WHERE
          IS_DELINQ = 1
    GROUP BY
          SK_ID_BUREAU

)

SELECT
      A.*,
      B.BB_MONTHS_LAST_DELINQ
FROM
      table_3 A
LEFT JOIN
      table_4 B
ON
      A.SK_ID_BUREAU = B.SK_ID_BUREAU
"""

BUREAU_BALANCE_JOIN_SQL = """
WITH table_1 AS (
    SELECT
          A.*,
          B.SK_ID_CURR
    FROM
          bureau_balance_df_preprocess A
    LEFT JOIN
          bureau_df B
    ON
          A.SK_ID_BUREAU = B.SK_ID_BUREAU
),

table_2 AS (
    SELECT
        SK_ID_CURR,
        AVG(BB_DELINQ_SHARE) AS BB_DELINQ_SHARE_MEAN,
        MAX(BB_WORST_STATUS_NUM) AS BB_WORST_STATUS_MAX,
        MIN(BB_MONTHS_LAST_DELINQ) AS BB_MONTHS_SINCE_LAST_DELINQ_MIN
    FROM
        table_1
    GROUP BY
        SK_ID_CURR
)

SELECT * FROM table_2
"""

BUREAU_FINAL_SQL = """
SELECT
      A.*,
      B.BB_DELINQ_SHARE_MEAN,
      B.BB_WORST_STATUS_MAX,
      B.BB_MONTHS_SINCE_LAST_DELINQ_MIN
FROM
      bureau_df_preprocess A
LEFT JOIN
      bureau_balance_join_df B
ON
      A.SK_ID_CURR = B.SK_ID_CURR
"""

PREVIOUS_APPLICATION_SQL = """
SELECT
      SK_ID_CURR,
      COUNT(DISTINCT SK_ID_PREV) AS PREV_TOTAL,

      SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS PREV_APPROVED,
      SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS PREV_REFUSED,
      SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1 ELSE 0 END) AS PREV_CANCELLED,
      SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Unused offer' THEN 1 ELSE 0 END) AS PREV_UNUSED,

      AVG(AMT_APPLICATION/AMT_CREDIT) AS PREV_APP_CREDIT_RATIO_MEAN,
      AVG(AMT_DOWN_PAYMENT/AMT_CREDIT) AS PREV_DOWNPAYMENT_RATE_MEAN,

      AVG(CNT_PAYMENT) AS PREV_CNT_PAYMENT_MEAN,
      MAX(DAYS_DECISION) AS PREV_LAST_DECISION_DAYS,
      MIN(DAYS_DECISION) AS PREV_EARLIEST_DECISION_DAYS,

      SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)/COUNT(DISTINCT SK_ID_PREV) AS PREV_APPROVAL_RATE,
      SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)/COUNT(DISTINCT SK_ID_PREV) AS PREV_REFUSED_RATE

FROM
    prev_application_df
GROUP BY
    SK_ID_CURR
"""

D_COLS = [
    "SK_ID_CURR",
    "INSTAL_PAYMENT_RATIO_MEAN",
    "INSTAL_PAYMENT_RATIO_STD",
    "INSTAL_LATE_MEAN",
    "INSTAL_LATE_MAX",
    "INSTAL_LATE_SHARE",
    "INSTAL_SEVERE_LATE_SHARE",
]

E_COLS = [
    "SK_ID_CURR",
    "CC_UTIL_MEAN",
    "CC_UTIL_MAX",
    "CC_MINPAY_COVERAGE_MEAN",
    "CC_MINPAY_MET_SHARE",
    "CC_DPD_ANY_SHARE",
    "CC_DPD_MAX",
    "CC_CASH_RATIO_MEAN",
]

F_COLS = [
    "SK_ID_CURR",
    "POS_DPD_ANY_SHARE",
    "POS_DPD_MAX",
    "POS_MONTHS_SINCE_LAST_ACTIVE",
]

INSTALLMENTS_SQL = """
WITH table_1 AS (
    SELECT
        *,
        AMT_PAYMENT / (AMT_INSTALMENT + {EPS}) AS PAYMENT_RATIO,
        DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT AS LATE_DAYS
    FROM
        installments_payments_df
),
table_2 AS (
    SELECT
        SK_ID_PREV,
        AVG(PAYMENT_RATIO) AS INSTAL_PAYMENT_RATIO_MEAN,
        STDDEV_SAMP(PAYMENT_RATIO) AS INSTAL_PAYMENT_RATIO_STD,
        AVG(LATE_DAYS) AS INSTAL_LATE_MEAN,
        MAX(LATE_DAYS) AS INSTAL_LATE_MAX,
        AVG(CASE WHEN LATE_DAYS > 0 THEN 1 ELSE 0 END) AS INSTAL_LATE_SHARE,
        AVG(CASE WHEN LATE_DAYS > 0 THEN 30 ELSE 0 END) AS INSTAL_SEVERE_LATE_SHARE
    FROM
        table_1
    GROUP BY
        SK_ID_PREV
),
table_3 AS (
    SELECT
        p.SK_ID_CURR,
        AVG(t2.INSTAL_PAYMENT_RATIO_MEAN) AS INSTAL_PAYMENT_RATIO_MEAN,
        AVG(t2.INSTAL_PAYMENT_RATIO_STD) AS INSTAL_PAYMENT_RATIO_STD,
        AVG(t2.INSTAL_LATE_MEAN) AS INSTAL_LATE_MEAN,
        MAX(t2.INSTAL_LATE_MAX) AS INSTAL_LATE_MAX,
        AVG(t2.INSTAL_LATE_SHARE) AS INSTAL_LATE_SHARE,
        AVG(t2.INSTAL_SEVERE_LATE_SHARE) AS INSTAL_SEVERE_LATE_SHARE
    FROM
        table_2 t2
    JOIN
        prev_application_df p
        ON t2.SK_ID_PREV = p.SK_ID_PREV
    GROUP BY
        p.SK_ID_CURR
)
SELECT * FROM table_3
"""

CREDIT_CARD_BALANCE_SQL = """
WITH table_1 AS (
    SELECT
        *,
        AMT_BALANCE / (AMT_CREDIT_LIMIT_ACTUAL + {EPS}) AS CC_UTIL_ROW,
        AMT_PAYMENT_TOTAL_CURRENT / (AMT_INST_MIN_REGULARITY + {EPS}) AS CC_MINPAY_COVERAGE_ROW,
        CASE WHEN (SK_DPD > 0) OR (SK_DPD_DEF > 0) THEN 1 ELSE 0 END AS CC_DPD_ANY_ROW,
        AMT_DRAWINGS_ATM_CURRENT / (AMT_CREDIT_LIMIT_ACTUAL + {EPS}) AS CC_CASH_RATIO_ROW
    FROM
        credit_card_balance_df
),
table_2 AS (
    SELECT
        SK_ID_PREV,
        AVG(CC_UTIL_ROW) AS CC_UTIL_MEAN,
        MAX(CC_UTIL_ROW) AS CC_UTIL_MAX,
        AVG(CC_MINPAY_COVERAGE_ROW) AS CC_MINPAY_COVERAGE_MEAN,
        AVG(CASE WHEN CC_MINPAY_COVERAGE_ROW >= 1 THEN 1 ELSE 0 END) AS CC_MINPAY_MET_SHARE,
        AVG(CC_DPD_ANY_ROW) AS CC_DPD_ANY_SHARE,
        MAX(SK_DPD) AS CC_DPD_MAX,
        AVG(CC_CASH_RATIO_ROW) AS CC_CASH_RATIO_MEAN
    FROM
        table_1
    GROUP BY
        SK_ID_PREV
),
table_3 AS (
    SELECT
        p.SK_ID_CURR,
        AVG(t2.CC_UTIL_MEAN) AS CC_UTIL_MEAN,
        MAX(t2.CC_UTIL_MAX) AS CC_UTIL_MAX,
        AVG(t2.CC_MINPAY_COVERAGE_MEAN) AS CC_MINPAY_COVERAGE_MEAN,
        AVG(t2.CC_MINPAY_MET_SHARE) AS CC_MINPAY_MET_SHARE,
        AVG(t2.CC_DPD_ANY_SHARE) AS CC_DPD_ANY_SHARE,
        MAX(t2.CC_DPD_MAX) AS CC_DPD_MAX,
        AVG(t2.CC_CASH_RATIO_MEAN) AS CC_CASH_RATIO_MEAN
    FROM
        table_2 t2
    JOIN
        prev_application_df p
        ON t2.SK_ID_PREV = p.SK_ID_PREV
    GROUP BY
        p.SK_ID_CURR
)
SELECT * FROM table_3
"""

POS_CASH_BALANCE_SQL = """
WITH table_1 AS (
    SELECT
        *,
        CASE WHEN (SK_DPD > 0) OR (SK_DPD_DEF > 0) THEN 1 ELSE 0 END AS POS_DPD_ANY_ROW
    FROM
        pos_cash_balance_df
),
table_2 AS (
    SELECT
        SK_ID_PREV,
        AVG(POS_DPD_ANY_ROW) AS POS_DPD_ANY_SHARE,
        MAX(SK_DPD) AS POS_DPD_MAX,
        -MAX(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN MONTHS_BALANCE ELSE NULL END) AS POS_MONTHS_SINCE_LAST_ACTIVE
    FROM
        table_1
    GROUP BY
        SK_ID_PREV
),
table_3 AS (
    SELECT
        p.SK_ID_CURR,
        AVG(t2.POS_DPD_ANY_SHARE) AS POS_DPD_ANY_SHARE,
        MAX(t2.POS_DPD_MAX) AS POS_DPD_MAX,
        MAX(t2.POS_MONTHS_SINCE_LAST_ACTIVE) AS POS_MONTHS_SINCE_LAST_ACTIVE
    FROM
        table_2 t2
    JOIN
        prev_application_df p
        ON t2.SK_ID_PREV = p.SK_ID_PREV
    GROUP BY
        p.SK_ID_CURR
)
SELECT * FROM table_3
"""
B_COLS = [
    "SK_ID_CURR",
    "BUREAU_N_ACTIVE",
    "BUREAU_N_CLOSED",
    "BUREAU_LAST_CREDIT_DAYS",
    "BUREAU_LAST_UPDATE_DAYS",
    "BUREAU_TOTAL_DEBT",
    "BUREAU_TOTAL_CREDIT",
    "BUREAU_LIMIT_SUM",
    "BUREAU_UTIL_MEAN",
    "BUREAU_OVERDUE_SUM",
    "BUREAU_MAX_OVERDUE",
    "BUREAU_PROLONG_SUM",
    "BB_DELINQ_SHARE_MEAN",
    "BB_WORST_STATUS_MAX",
    "BB_MONTHS_SINCE_LAST_DELINQ_MIN",
]

C_COLS = [
    "SK_ID_CURR",
    "PREV_TOTAL",
    "PREV_APPROVED",
    "PREV_REFUSED",
    "PREV_CANCELLED",
    "PREV_UNUSED",
    "PREV_APP_CREDIT_RATIO_MEAN",
    "PREV_DOWNPAYMENT_RATE_MEAN",
    "PREV_CNT_PAYMENT_MEAN",
    "PREV_LAST_DECISION_DAYS",
    "PREV_EARLIEST_DECISION_DAYS",
    "PREV_APPROVAL_RATE",
    "PREV_REFUSED_RATE",
]


@dataclass
class SqlFeatureStoreInputs:
    application_df: pd.DataFrame
    bureau_df: pd.DataFrame
    bureau_balance_df: pd.DataFrame
    prev_application_df: pd.DataFrame
    installments_payments_df: pd.DataFrame
    credit_card_balance_df: pd.DataFrame
    pos_cash_balance_df: pd.DataFrame


def _register_frames(con: duckdb.DuckDBPyConnection, frames: Dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        con.register(name, frame)


def _run_sql(con: duckdb.DuckDBPyConnection, sql: str, eps: float) -> pd.DataFrame:
    return con.execute(sql.format(EPS=eps)).df()


def build_feature_store_via_sql(
    inputs: SqlFeatureStoreInputs,
    eps: float = EPS_DEFAULT,
) -> pd.DataFrame:
    """Execute the verbatim notebook SQL to construct the feature store."""
    con = duckdb.connect(database=":memory:")
    try:
        _register_frames(
            con,
            {
                "application_df": inputs.application_df,
                "bureau_df": inputs.bureau_df,
                "bureau_balance_df": inputs.bureau_balance_df,
                "prev_application_df": inputs.prev_application_df,
                "installments_payments_df": inputs.installments_payments_df,
                "credit_card_balance_df": inputs.credit_card_balance_df,
                "pos_cash_balance_df": inputs.pos_cash_balance_df,
            },
        )

        application_df_preprocess = _run_sql(con, APPLICATION_SQL, eps)
        con.register("application_df_preprocess", application_df_preprocess)

        bureau_df_preprocess = _run_sql(con, BUREAU_SQL, eps)
        con.register("bureau_df_preprocess", bureau_df_preprocess)

        bureau_balance_df_preprocess = _run_sql(con, BUREAU_BALANCE_SQL, eps)
        con.register("bureau_balance_df_preprocess", bureau_balance_df_preprocess)

        bureau_balance_join_df = con.execute(BUREAU_BALANCE_JOIN_SQL).df()
        con.register("bureau_balance_join_df", bureau_balance_join_df)

        bureau_df_preprocess_2 = con.execute(BUREAU_FINAL_SQL).df()
        con.register("bureau_df_preproces_2", bureau_df_preprocess_2)

        prev_application_df_preprocess = con.execute(PREVIOUS_APPLICATION_SQL).df()
        installments_df_preprocess = _run_sql(con, INSTALLMENTS_SQL, eps)
        creditcard_balance_df_preprocess = _run_sql(con, CREDIT_CARD_BALANCE_SQL, eps)
        poscash_balance_df_preprocess = _run_sql(con, POS_CASH_BALANCE_SQL, eps)

    finally:
        con.close()

    feature_store_df = (
        application_df_preprocess.merge(
            bureau_df_preprocess_2[B_COLS],
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        ).merge(
            prev_application_df_preprocess[C_COLS],
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        ).merge(
            installments_df_preprocess[D_COLS],
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        ).merge(
            creditcard_balance_df_preprocess[E_COLS],
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        ).merge(
            poscash_balance_df_preprocess[F_COLS],
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )
    )

    categorical_columns = feature_store_df.select_dtypes(include=["object"]).columns
    feature_store_df = pd.get_dummies(feature_store_df, columns=categorical_columns)
    feature_store_df = feature_store_df.fillna(feature_store_df.median(numeric_only=True))

    return feature_store_df
