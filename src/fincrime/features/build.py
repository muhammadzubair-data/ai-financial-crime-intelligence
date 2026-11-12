"""Transaction features, built in DuckDB.

Every behavioural feature looks strictly backwards in time from the transaction being scored,
so nothing here uses information that would not exist at the moment the transaction happens.

The trick that keeps this simple is the *ledger*: each transaction is written once for the
account that pays (debit) and once for the account that receives (credit). Rolling windows are
computed per account on the ledger, then joined back twice, giving sender-side features
(prefix ``s_``) and receiver-side features (prefix ``d_``).
"""
from __future__ import annotations

from pathlib import Path

from .. import config as C

CATEGORICAL = ["txn_type", "channel", "s_type", "d_type", "s_kyc", "d_kyc"]

SIDE = ["n_24h", "n_7d", "cr_sum_24h", "db_sum_24h", "cr_sum_72h", "db_sum_72h", "cr_cnt_7d", "db_cnt_7d",
        "senders_7d", "receivers_7d", "sub10k_cash_7d", "avg_30d", "std_30d", "n_30d", "secs_since_prev",
        "secs_since_credit", "n_prior", "dev_share"]

NUMERIC = (["amount", "log_amount", "hour", "dow", "is_night", "is_weekend", "is_international", "is_high_risk_geo",
            "just_below_threshold", "is_round", "pair_prior", "dev_accounts", "s_new_device", "s_passthrough",
            "s_out_in_72h", "s_amount_z", "s_amount_to_income", "s_age_days", "d_age_days", "d_amount_z",
            "d_amount_to_income"]
           + [f"s_{c}" for c in SIDE] + [f"d_{c}" for c in SIDE])
FEATURES = NUMERIC + CATEGORICAL

FRAME = "RANGE BETWEEN INTERVAL {} PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW"


def build_features(data_dir: Path | None = None, threads: int | None = None) -> Path:
    import duckdb      # imported here so the Streamlit app can read the feature list without DuckDB installed

    data_dir = Path(data_dir or C.DATA_DIR)
    out = data_dir / "features.parquet"
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads={int(threads)}")
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet('{data_dir / 'transactions.parquet'}')")
    con.execute(f"CREATE VIEW acct AS SELECT * FROM read_parquet('{data_dir / 'accounts.parquet'}')")

    # ---- ledger: one row per (transaction, internal account involved)
    con.execute("""
        CREATE TABLE ledger AS
        SELECT txn_id, ts, src_account AS account, 'D' AS dir, amount, txn_type,
               CASE WHEN dst_account >= 0 THEN dst_account WHEN ext_party >= 0 THEN -ext_party - 2 END AS cp
        FROM tx WHERE src_account >= 0
        UNION ALL
        SELECT txn_id, ts, dst_account, 'C', amount, txn_type,
               CASE WHEN src_account >= 0 THEN src_account WHEN ext_party >= 0 THEN -ext_party - 2 END
        FROM tx WHERE dst_account >= 0
    """)

    # ---- devices: how many different accounts has this device been used by, up to now
    con.execute("""
        CREATE TABLE dev AS
        WITH d AS (
            SELECT txn_id, ts, device_id, CASE WHEN src_account >= 0 THEN src_account ELSE dst_account END AS account
            FROM tx WHERE device_id >= 0),
        f AS (
            SELECT *, (row_number() OVER (PARTITION BY device_id, account ORDER BY ts, txn_id) = 1)::INT AS first_pair
            FROM d)
        SELECT txn_id, ts, account,
               sum(first_pair) OVER (PARTITION BY device_id ORDER BY ts, txn_id) AS dev_accounts,
               (first_pair = 1 AND row_number() OVER (PARTITION BY account ORDER BY ts, txn_id) > 1)::INT AS new_device
        FROM f
    """)
    # widest device sharing seen on each account so far (looked up "as of" each ledger row)
    con.execute("""
        CREATE TABLE acct_dev AS
        SELECT account, ts, max(dev_accounts) OVER (PARTITION BY account ORDER BY ts, txn_id) AS dev_share FROM dev
    """)

    w = {k: FRAME.format(v) for k, v in {"h24": "24 HOURS", "h72": "72 HOURS", "d7": "7 DAYS", "d30": "30 DAYS"}.items()}
    con.execute(f"""
        CREATE TABLE lf AS
        SELECT l.txn_id, l.account, l.dir,
            count(*) OVER w24 AS n_24h,
            count(*) OVER w7 AS n_7d,
            coalesce(sum(CASE WHEN dir = 'C' THEN amount END) OVER w24, 0) AS cr_sum_24h,
            coalesce(sum(CASE WHEN dir = 'D' THEN amount END) OVER w24, 0) AS db_sum_24h,
            coalesce(sum(CASE WHEN dir = 'C' THEN amount END) OVER w72, 0) AS cr_sum_72h,
            coalesce(sum(CASE WHEN dir = 'D' THEN amount END) OVER w72, 0) AS db_sum_72h,
            count(CASE WHEN dir = 'C' THEN 1 END) OVER w7 AS cr_cnt_7d,
            count(CASE WHEN dir = 'D' THEN 1 END) OVER w7 AS db_cnt_7d,
            count(DISTINCT CASE WHEN dir = 'C' THEN cp END) OVER w7 AS senders_7d,
            count(DISTINCT CASE WHEN dir = 'D' THEN cp END) OVER w7 AS receivers_7d,
            count(CASE WHEN txn_type = 'cash_deposit' AND amount >= {C.STRUCT_LOW}
                        AND amount < {C.CASH_REPORT_THRESHOLD} THEN 1 END) OVER w7 AS sub10k_cash_7d,
            avg(amount) OVER w30 AS avg_30d,
            stddev_samp(amount) OVER w30 AS std_30d,
            count(*) OVER w30 AS n_30d,
            epoch(ts - lag(ts) OVER w) AS secs_since_prev,
            epoch(ts - max(CASE WHEN dir = 'C' THEN ts END) OVER wprev) AS secs_since_credit,
            row_number() OVER w - 1 AS n_prior
        FROM ledger l
        WINDOW w AS (PARTITION BY account ORDER BY ts, txn_id),
               wprev AS (PARTITION BY account ORDER BY ts, txn_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
               w24 AS (PARTITION BY account ORDER BY ts {w['h24']}),
               w72 AS (PARTITION BY account ORDER BY ts {w['h72']}),
               w7 AS (PARTITION BY account ORDER BY ts {w['d7']}),
               w30 AS (PARTITION BY account ORDER BY ts {w['d30']})
    """)
    con.execute("""
        CREATE TABLE lf2 AS
        SELECT lf.*, coalesce(a.dev_share, 1) AS dev_share
        FROM (SELECT lf.*, l.ts FROM lf JOIN ledger l USING (txn_id, account, dir)) lf
        ASOF LEFT JOIN acct_dev a ON lf.account = a.account AND lf.ts >= a.ts
    """)

    side = lambda p: ", ".join(f"{p}.{c} AS {p}_{c}" for c in SIDE)  # noqa: E731
    hr = ", ".join(f"'{c}'" for c in C.HIGH_RISK)
    con.execute(f"""
        COPY (
        SELECT t.txn_id, t.ts, t.src_account, t.dst_account, t.is_suspicious, t.typology, t.network_id,
            t.txn_type, t.channel, t.amount,
            ln(1 + t.amount) AS log_amount,
            hour(t.ts) AS hour, dayofweek(t.ts) AS dow,
            (hour(t.ts) < 6 OR hour(t.ts) >= 22)::INT AS is_night,
            (dayofweek(t.ts) IN (0, 6))::INT AS is_weekend,
            (t.ext_country <> '{C.DOMESTIC}')::INT AS is_international,
            (t.ext_country IN ({hr}))::INT AS is_high_risk_geo,
            (t.amount >= {C.STRUCT_LOW} AND t.amount < {C.CASH_REPORT_THRESHOLD})::INT AS just_below_threshold,
            (t.amount >= 500 AND t.amount % 100 = 0)::INT AS is_round,
            row_number() OVER (PARTITION BY t.src_account, t.dst_account, t.ext_party ORDER BY t.ts, t.txn_id) - 1 AS pair_prior,
            dv.dev_accounts, dv.new_device AS s_new_device,
            least(t.amount / nullif(s.cr_sum_24h, 0), 20) AS s_passthrough,
            least((s.db_sum_72h + t.amount) / nullif(s.cr_sum_72h, 0), 20) AS s_out_in_72h,
            (t.amount - s.avg_30d) / nullif(s.std_30d, 0) AS s_amount_z,
            t.amount / (sa.annual_income / 12) AS s_amount_to_income,
            date_diff('day', sa.open_date, t.ts) AS s_age_days,
            date_diff('day', da.open_date, t.ts) AS d_age_days,
            (t.amount - d.avg_30d) / nullif(d.std_30d, 0) AS d_amount_z,
            t.amount / (da.annual_income / 12) AS d_amount_to_income,
            {side('s')}, {side('d')},
            sa.customer_type AS s_type, da.customer_type AS d_type, sa.kyc_risk AS s_kyc, da.kyc_risk AS d_kyc
        FROM tx t
        LEFT JOIN lf2 s ON s.txn_id = t.txn_id AND s.dir = 'D'
        LEFT JOIN lf2 d ON d.txn_id = t.txn_id AND d.dir = 'C'
        LEFT JOIN dev dv ON dv.txn_id = t.txn_id
        LEFT JOIN acct sa ON sa.account_id = t.src_account
        LEFT JOIN acct da ON da.account_id = t.dst_account
        ORDER BY t.txn_id
        ) TO '{out}' (FORMAT PARQUET)
    """)
    con.close()
    return out


if __name__ == "__main__":
    print(build_features())
