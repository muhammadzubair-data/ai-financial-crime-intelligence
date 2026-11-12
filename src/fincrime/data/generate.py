"""Synthetic bank data generator.

Produces four parquet files in the data directory:

    accounts.parquet          one row per customer account
    transactions.parquet      one row per transaction (12 months)
    networks.parquet          ground truth: one row per injected suspicious network
    network_members.parquet   ground truth: accounts in each network and their role

EVERYTHING HERE IS SYNTHETIC. No real customer, bank or transaction is represented.

Legitimate activity deliberately includes behaviour that looks like the typologies
(cash-heavy businesses depositing just under 10,000, payroll fan-out, marketplace sellers
with many first-time senders, households sharing a device) so the detection problem is
not trivial. One in five networks is "stealth": smaller amounts and much slower movement.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C

DAY = 86_400
HOUR_P = np.array([1, 1, 1, 1, 1, 2, 4, 7, 9, 10, 10, 10, 10, 9, 9, 9, 9, 9, 8, 7, 5, 4, 2, 1], dtype=float)
HOUR_P /= HOUR_P.sum()
NIGHT_P = np.array([8, 8, 8, 7, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8], dtype=float)
NIGHT_P /= NIGHT_P.sum()

# external counterparties share one id space
N_MERCHANT, N_BILLER, N_EMPLOYER, N_EXTBANK = 5_000, 300, 2_000, 20_000
OFF_BILLER = N_MERCHANT
OFF_EMPLOYER = OFF_BILLER + N_BILLER
OFF_EXTBANK = OFF_EMPLOYER + N_EMPLOYER

COLS = ["ts", "txn_type", "channel", "amount", "src_account", "dst_account", "ext_party", "ext_country",
        "device_id", "is_suspicious", "typology", "network_id"]


def _frame(ts, txn_type, channel, amount, src=-1, dst=-1, ext=-1, country=C.DOMESTIC, device=-1):
    n = len(ts)

    def col(v, dtype=None):
        return np.full(n, v, dtype=dtype) if np.isscalar(v) else np.asarray(v)

    return pd.DataFrame({
        "ts": np.asarray(ts, dtype=np.int64), "txn_type": col(txn_type), "channel": col(channel),
        "amount": np.round(np.asarray(amount, dtype=float), 2),
        "src_account": col(src, np.int64), "dst_account": col(dst, np.int64),
        "ext_party": col(ext, np.int64), "ext_country": col(country), "device_id": col(device, np.int64),
        "is_suspicious": 0, "typology": "none", "network_id": -1,
    })


class Generator:
    def __init__(self, scale: C.Scale, seed: int = C.SEED):
        self.s = scale
        self.rng = np.random.default_rng(seed)
        self.n = scale.n_accounts

    # ------------------------------------------------------------------ accounts
    def make_accounts(self) -> None:
        rng, n = self.rng, self.n
        business = rng.random(n) < 0.12
        income = np.where(business, rng.lognormal(np.log(600_000), 0.9, n), rng.lognormal(np.log(52_000), 0.5, n))
        # most accounts are years old; 8% were opened during the year
        open_day = -rng.integers(30, 3650, n)
        new = rng.random(n) < 0.08
        open_day[new] = rng.integers(0, 330, new.sum())
        activity = rng.lognormal(0, 0.8, n)
        activity /= activity.mean()
        dev1 = np.arange(n)
        has2 = rng.random(n) < 0.35
        dev2 = np.where(has2, n + np.arange(n), -1)
        # households: about 3% of accounts share their main device with the next account
        hh = np.flatnonzero(rng.random(n - 1) < 0.03)
        dev1[hh + 1] = dev1[hh]
        # personal contacts: mostly "nearby" accounts so friend groups overlap
        k = rng.integers(2, 7, n)
        near = (np.arange(n)[:, None] + rng.integers(-40, 41, (n, 6))) % n
        far = rng.integers(0, n, (n, 6))
        contacts = np.where(rng.random((n, 6)) < 0.7, near, far)
        contacts = np.where(contacts == np.arange(n)[:, None], (contacts + 1) % n, contacts)

        self.business, self.income, self.open_day, self.activity = business, income, open_day, activity
        self.dev1, self.dev2, self.k, self.contacts = dev1, dev2, k, contacts
        self.kyc = rng.choice(["low", "medium", "high"], n, p=[0.78, 0.18, 0.04])
        self.segment = np.where(business, rng.choice(["retail_trade", "services", "hospitality", "wholesale", "other"], n),
                                rng.choice(["mass", "affluent", "student", "senior"], n, p=[0.6, 0.15, 0.1, 0.15]))
        self.reserved = np.zeros(n, dtype=bool)      # accounts given a special legitimate or suspicious role
        biz = np.flatnonzero(business & (open_day < 0))
        rng.shuffle(biz)
        n_cash, n_payroll = max(2, len(biz) // 20), max(2, len(biz) // 30)
        self.cash_biz, self.payroll_biz = biz[:n_cash], biz[n_cash:n_cash + n_payroll]
        self.old_retail = np.flatnonzero(~business & (open_day < 0))
        self.sellers = rng.choice(self.old_retail, max(3, n // 150), replace=False)
        self.reserved[np.concatenate([self.cash_biz, self.payroll_biz, self.sellers])] = True
        self.next_device = 2 * n                     # ids above this are devices controlled by a network

    # ------------------------------------------------------------------ helpers
    def _days(self, n, acct=None, p=None):
        """Random day in the year; accounts opened during the year only transact after opening."""
        d = self.rng.random(n) * C.N_DAYS
        if acct is not None:
            od = np.maximum(self.open_day[acct], 0)
            d = od + d * (C.N_DAYS - od) / C.N_DAYS
        return d

    def _ts(self, n, acct=None, hour_p=HOUR_P):
        d = np.floor(self._days(n, acct)).astype(np.int64)
        return d * DAY + self.rng.choice(24, n, p=hour_p) * 3600 + self.rng.integers(0, 3600, n)

    def _device(self, acct):
        use2 = (self.dev2[acct] >= 0) & (self.rng.random(len(acct)) < 0.2)
        return np.where(use2, self.dev2[acct], self.dev1[acct])

    def _expand(self, rate):
        """Draw a Poisson count per account and return the repeated account index."""
        return np.repeat(np.arange(self.n), self.rng.poisson(rate))

    def _ext_bank(self, n, foreign_share=0.15, high_risk_share=0.0):
        rng = self.rng
        ids = OFF_EXTBANK + rng.integers(0, N_EXTBANK, n)
        country = np.full(n, C.DOMESTIC, dtype=object)
        u = rng.random(n)
        low_risk = [c for c in C.FOREIGN if c not in C.HIGH_RISK]
        f = u < foreign_share
        country[f] = rng.choice(low_risk, f.sum())
        h = u < high_risk_share
        country[h] = rng.choice(C.HIGH_RISK, h.sum())
        return ids, country

    # ------------------------------------------------------------------ legitimate activity
    def legit(self) -> pd.DataFrame:
        rng, n, b = self.rng, self.n, self.business
        inc = self.income / np.where(b, 600_000, 52_000)             # relative spending power
        # yearly transaction rates per account by type: (retail, business)
        rates = {"card": (4.6, 3.0), "bill": (1.3, 4.0), "p2p": (1.6, 3.0), "cash_dep": (0.35, 5.0),
                 "cash_wd": (0.8, 1.0), "wire_out": (0.05, 1.2), "wire_in": (0.04, 1.2), "credit": (0.6, 14.0)}
        salaried = (~b) & (rng.random(n) < 0.30)
        expected = sum(np.where(b, rb, rr).mean() for rr, rb in rates.values()) * n + salaried.sum() * 12
        mult = self.s.target_txns / expected
        parts = []

        def rate(key):
            rr, rb = rates[key]
            return np.where(b, rb, rr) * self.activity * mult

        a = self._expand(rate("card"))
        parts.append(_frame(self._ts(len(a), a), "card_payment", rng.choice(["pos", "online"], len(a), p=[0.65, 0.35]),
                            rng.lognormal(3.6, 0.9, len(a)) * inc[a] ** 0.5, src=a, ext=rng.integers(0, N_MERCHANT, len(a))))
        parts[-1]["device_id"] = np.where(parts[-1]["channel"] == "online", self._device(a), -1)

        a = self._expand(rate("bill"))
        parts.append(_frame(self._ts(len(a), a), "bill_payment", rng.choice(["mobile", "web"], len(a), p=[0.6, 0.4]),
                            rng.lognormal(4.6, 0.6, len(a)) * inc[a] ** 0.5, src=a,
                            ext=OFF_BILLER + rng.integers(0, N_BILLER, len(a)), device=self._device(a)))

        a = self._expand(rate("p2p"))
        dst = self.contacts[a, rng.integers(0, 6, len(a)) % self.k[a]]
        parts.append(_frame(self._ts(len(a), a), "transfer", rng.choice(["mobile", "web"], len(a), p=[0.75, 0.25]),
                            np.minimum(rng.lognormal(4.4, 1.1, len(a)) * inc[a] ** 0.5, 25_000), src=a, dst=dst,
                            device=self._device(a)))

        a = self._expand(rate("cash_dep"))
        amt = np.where(b[a], rng.lognormal(7.6, 0.9, len(a)), rng.lognormal(5.5, 0.9, len(a)))
        parts.append(_frame(self._ts(len(a), a), "cash_deposit", rng.choice(["branch", "atm"], len(a), p=[0.6, 0.4]), amt, dst=a))

        a = self._expand(rate("cash_wd"))
        parts.append(_frame(self._ts(len(a), a), "cash_withdrawal", rng.choice(["atm", "branch"], len(a), p=[0.85, 0.15]),
                            rng.lognormal(5.0, 0.7, len(a)), src=a))

        a = self._expand(rate("wire_out"))
        ext, ctry = self._ext_bank(len(a), 0.55, 0.03)
        parts.append(_frame(self._ts(len(a), a), "wire_out", rng.choice(["web", "branch"], len(a), p=[0.7, 0.3]),
                            rng.lognormal(7.5, 1.0, len(a)) * inc[a] ** 0.5, src=a, ext=ext, country=ctry))
        parts[-1]["device_id"] = np.where(parts[-1]["channel"] == "web", self._device(a), -1)

        a = self._expand(rate("wire_in"))
        ext, ctry = self._ext_bank(len(a), 0.55, 0.03)
        parts.append(_frame(self._ts(len(a), a), "wire_in", "wire", rng.lognormal(7.5, 1.0, len(a)) * inc[a] ** 0.5,
                            dst=a, ext=ext, country=ctry))

        a = self._expand(rate("credit"))       # customer payments, refunds, benefits
        amt = np.where(b[a], rng.lognormal(6.8, 1.1, len(a)), rng.lognormal(5.2, 0.9, len(a)))
        parts.append(_frame(self._ts(len(a), a), "incoming_transfer", "ach", amt, dst=a,
                            ext=OFF_EXTBANK + rng.integers(0, N_EXTBANK, len(a))))

        # monthly salary, and for most salaried customers a rent payment shortly afterwards
        acc = np.flatnonzero(salaried)
        employer = OFF_EMPLOYER + rng.integers(0, N_EMPLOYER, len(acc))
        payday = rng.integers(22, 29, len(acc))
        pays_rent = rng.random(len(acc)) < 0.6
        landlord = self.contacts[acc, 0]
        for m in range(12):
            day = (pd.Timestamp(2025, m + 1, 1) - C.START).days + payday
            ok = day >= self.open_day[acc]
            t = day[ok] * DAY + rng.integers(1, 7, ok.sum()) * 3600 + rng.integers(0, 3600, ok.sum())
            sal = self.income[acc][ok] / 12 * rng.normal(0.72, 0.02, ok.sum())
            parts.append(_frame(t, "incoming_transfer", "ach", sal, dst=acc[ok], ext=employer[ok]))
            r = ok & pays_rent
            t = day[r] * DAY + rng.integers(DAY, 6 * DAY, r.sum())
            parts.append(_frame(t, "transfer", "mobile", self.income[acc][r] / 12 * rng.uniform(0.28, 0.42, r.sum()),
                                src=acc[r], dst=landlord[r], device=self.dev1[acc[r]]))

        parts += self._lookalikes()
        df = pd.concat(parts, ignore_index=True)
        df = df[(df.ts >= 0) & (df.ts < C.N_DAYS * DAY) & (df.amount >= 1)]
        # a transfer cannot arrive before the receiving account exists
        d = df.dst_account.to_numpy()
        ok = (d < 0) | (self.open_day[np.maximum(d, 0)] * DAY <= df.ts.to_numpy())
        return df[ok]

    def _lookalikes(self) -> list[pd.DataFrame]:
        """Legitimate behaviour that resembles the suspicious typologies (hard negatives)."""
        rng, n = self.rng, self.n
        out = []
        cash_biz, payroll_biz, sellers, retail = self.cash_biz, self.payroll_biz, self.sellers, self.old_retail
        contact = lambda a: self.contacts[a, rng.integers(0, 6, len(a)) % self.k[a]]  # noqa: E731
        some = lambda share: rng.choice(retail, max(2, int(len(retail) * share)), replace=False)  # noqa: E731

        # cash-intensive businesses: frequent deposits, many just under the reporting threshold
        a = np.repeat(cash_biz, rng.integers(25, 110, len(cash_biz)))
        out.append(_frame(self._ts(len(a)), "cash_deposit", "branch", rng.uniform(5_500, 9_950, len(a)), dst=a))
        # payroll: one business pays 5-40 employees on the same day every month
        for emp in payroll_biz:
            staff = rng.choice(retail, rng.integers(5, 41), replace=False)
            wage = rng.uniform(1_500, 6_000, len(staff))
            for m in range(12):
                day = (pd.Timestamp(2025, m + 1, 1) - C.START).days + 25
                t = day * DAY + 9 * 3600 + rng.integers(0, 1800, len(staff))
                out.append(_frame(t, "transfer", "web", wage * rng.normal(1, 0.01, len(staff)), src=emp, dst=staff,
                                  device=self.dev1[emp]))
        # marketplace sellers: many small payments from strangers
        a = np.repeat(sellers, rng.integers(20, 150, len(sellers)))
        buyers = rng.integers(0, n, len(a))
        buyers = np.where(buyers == a, (buyers + 1) % n, buyers)
        out.append(_frame(self._ts(len(a), buyers), "transfer", "mobile", rng.lognormal(4.3, 0.8, len(a)), src=buyers,
                          dst=a, device=self.dev1[buyers]))

        # group organisers (club treasurers, flat-shares): collect from several people, then pay it all out
        org = np.repeat(some(0.015), rng.integers(2, 9, max(2, int(len(retail) * 0.015))))
        t0 = self._ts(len(org))
        e = np.repeat(np.arange(len(org)), rng.integers(3, 11, len(org)))
        payer = np.where(rng.random(len(e)) < 0.5, contact(org[e]), rng.integers(0, n, len(e)))
        payer = np.where(payer == org[e], (payer + 1) % n, payer)
        amt = np.clip(rng.lognormal(5.3, 0.8, len(e)), 20, 2_500)
        out.append(_frame(t0[e] + rng.integers(0, 2 * DAY, len(e)), "transfer", "mobile", amt, src=payer, dst=org[e],
                          device=self.dev1[payer]))
        total = np.bincount(e, amt, minlength=len(org)) * rng.uniform(0.85, 1.0, len(org))
        pay_t = t0 + 2 * DAY + rng.integers(3600, 36 * 3600, len(org))
        to_bill = rng.random(len(org)) < 0.5
        out.append(_frame(pay_t[to_bill], "bill_payment", "web", total[to_bill], src=org[to_bill],
                          ext=OFF_BILLER + rng.integers(0, N_BILLER, to_bill.sum()), device=self.dev1[org[to_bill]]))
        out.append(_frame(pay_t[~to_bill], "transfer", "mobile", total[~to_bill], src=org[~to_bill],
                          dst=contact(org[~to_bill]), device=self.dev1[org[~to_bill]]))

        # big-ticket life events (house, car): a large sum arrives and most of it leaves within days
        a = rng.choice(np.flatnonzero(self.open_day < 0), max(2, n // 50), replace=False)
        t = self._ts(len(a))
        amt = np.clip(rng.lognormal(10.3, 0.8, len(a)), 8_000, 400_000)
        ext, ctry = self._ext_bank(len(a), 0.2, 0.02)
        by_wire = rng.random(len(a)) < 0.5
        out.append(_frame(t[by_wire], "wire_in", "wire", amt[by_wire], dst=a[by_wire], ext=ext[by_wire], country=ctry[by_wire]))
        out.append(_frame(t[~by_wire], "incoming_transfer", "ach", amt[~by_wire], dst=a[~by_wire], ext=ext[~by_wire]))
        t2 = t + (rng.uniform(0.2, 5, len(a)) * DAY).astype(np.int64)
        ext, ctry = self._ext_bank(len(a), 0.2, 0.02)
        w = rng.random(len(a)) < 0.6
        out.append(_frame(t2[w], "wire_out", "web", amt[w] * rng.uniform(0.8, 1.0, w.sum()), src=a[w], ext=ext[w],
                          country=ctry[w], device=self.dev1[a[w]]))
        out.append(_frame(t2[~w], "transfer", "web", amt[~w] * rng.uniform(0.8, 1.0, (~w).sum()), src=a[~w],
                          dst=contact(a[~w]), device=self.dev1[a[~w]]))

        # remitters: a monthly wire to family abroad, half of them to a higher-risk jurisdiction
        a = some(0.015)
        ext, ctry = self._ext_bank(len(a), 1.0, 0.5)
        day = rng.integers(1, 28, len(a))
        for m in range(12):
            t = ((pd.Timestamp(2025, m + 1, 1) - C.START).days + day) * DAY + rng.integers(8, 20, len(a)) * 3600
            out.append(_frame(t, "wire_out", rng.choice(["web", "branch"], len(a)), rng.lognormal(6.3, 0.6, len(a)), src=a,
                              ext=ext, country=ctry, device=self.dev1[a]))

        # cash earners (tips, market stalls, private car sales): bursts of sizeable cash deposits
        a = np.repeat(some(0.01), rng.integers(2, 7, max(2, int(len(retail) * 0.01))))
        t0 = self._ts(len(a))
        e = np.repeat(np.arange(len(a)), rng.integers(2, 6, len(a)))
        amt = rng.uniform(2_500, 9_900, len(e))
        out.append(_frame(t0[e] + rng.integers(0, 6 * DAY, len(e)), "cash_deposit", rng.choice(["branch", "atm"], len(e)),
                          amt, dst=a[e]))
        mv = rng.random(len(a)) < 0.5
        total = np.bincount(e, amt, minlength=len(a))
        out.append(_frame(t0[mv] + 6 * DAY + rng.integers(0, 3 * DAY, mv.sum()), "transfer", "mobile",
                          total[mv] * rng.uniform(0.5, 0.95, mv.sum()), src=a[mv], dst=contact(a[mv]), device=self.dev1[a[mv]]))

        # family or flat-share loops: three people passing similar amounts round in a circle
        a = some(0.003)
        b_, c_ = self.contacts[a, 0], self.contacts[a, 1]
        keep = (b_ != c_) & (c_ != a)
        a, b_, c_ = a[keep], b_[keep], c_[keep]
        for _ in range(5):
            t = self._ts(len(a))
            amt = rng.uniform(1_000, 4_000, len(a))
            for src, dst in ((a, b_), (b_, c_), (c_, a)):
                out.append(_frame(t, "transfer", "mobile", amt, src=src, dst=dst, device=self.dev1[src]))
                t = t + rng.integers(2 * 3600, 40 * 3600, len(a))
                amt = amt * rng.uniform(0.9, 1.0, len(a))

        # newly opened accounts funded with a lump sum that is then spent quickly
        new = np.flatnonzero((self.open_day >= 0) & ~self.reserved)
        a = new[rng.random(len(new)) < 0.3]
        t = (self.open_day[a] * DAY + rng.uniform(0.5, 10, len(a)) * DAY).astype(np.int64)
        amt = rng.uniform(2_000, 40_000, len(a))
        out.append(_frame(t, "incoming_transfer", "ach", amt, dst=a, ext=OFF_EXTBANK + rng.integers(0, N_EXTBANK, len(a))))
        e = np.repeat(np.arange(len(a)), rng.integers(2, 7, len(a)))
        share = rng.dirichlet(np.ones(6), len(a))[e, rng.integers(0, 6, len(e))] * 1.2
        out.append(_frame(t[e] + rng.integers(3600, 5 * DAY, len(e)), "transfer", "mobile", amt[e] * share, src=a[e],
                          dst=contact(a[e]), device=self.dev1[a[e]]))
        return out

    # ------------------------------------------------------------------ suspicious networks
    def _pick(self, k, new_share=0.0, start_day=0):
        """Reserve k unused retail accounts; optionally make some of them recently opened."""
        k_biz = int(self.rng.binomial(k, 0.2))      # one member in five is a business (shell company)
        acc = np.concatenate([
            self.rng.choice(np.flatnonzero(~self.reserved & ~self.business), k - k_biz, replace=False),
            self.rng.choice(np.flatnonzero(~self.reserved & self.business), k_biz, replace=False)])
        self.rng.shuffle(acc)
        self.reserved[acc] = True
        recent = self.rng.random(k) < new_share
        self.open_day[acc[recent]] = start_day - self.rng.integers(3, 60, recent.sum())
        self.open_day[acc] = np.minimum(self.open_day[acc], start_day - 1)
        return acc

    def _victims(self, k):
        pool = np.flatnonzero(~self.reserved)
        return self.rng.choice(pool, k, replace=True)

    def _risky_ext(self, high_risk_p):
        ids, ctry = self._ext_bank(1, 1.0, high_risk_p)
        return int(ids[0]), ctry[0]

    def suspicious(self):
        rng, s = self.rng, self.s
        plan = (["mule_ring"] * s.n_mule_rings + ["circular"] * s.n_circular
                + ["structuring"] * s.n_structuring + ["layering"] * s.n_layering)
        rng.shuffle(plan)
        rows, nets, members = [], [], []
        for net_id, typ in enumerate(plan):
            stealth = bool(rng.random() < 0.20)
            self.rows, self.net_id, self.typ = [], net_id, typ
            roles = getattr(self, f"_net_{typ}")(stealth)
            ts = [r[0] for r in self.rows]
            nets.append((net_id, typ, stealth, len(roles), len(self.rows), min(ts), max(ts),
                         round(sum(r[3] for r in self.rows), 2)))
            members += [(net_id, int(a), role) for a, role in roles.items()]
            rows += self.rows
        tx = pd.DataFrame(rows, columns=COLS)
        tx = tx[(tx.ts >= 0) & (tx.ts < C.N_DAYS * DAY)]
        nets = pd.DataFrame(nets, columns=["network_id", "typology", "stealth", "n_members", "n_txns", "first_ts",
                                           "last_ts", "total_amount"])
        members = pd.DataFrame(members, columns=["network_id", "account_id", "role"])
        return tx, nets, members

    def _add(self, t, txn_type, channel, amount, src=-1, dst=-1, ext=-1, country=C.DOMESTIC, device=-1):
        self.rows.append((int(t), txn_type, channel, round(float(amount), 2), int(src), int(dst), int(ext), country,
                          int(device), 1, self.typ, self.net_id))

    def _start(self, duration_days):
        return float(self.rng.uniform(3, C.N_DAYS - duration_days - 6)) * DAY

    def _net_mule_ring(self, stealth):
        """Scam victims pay into mule accounts; mules forward to a collector who cashes out."""
        rng = self.rng
        dur = int(rng.integers(14, 70))
        t0 = self._start(dur)
        n_m = int(rng.integers(4, 13))
        mules = self._pick(n_m, new_share=0.45, start_day=int(t0 // DAY))
        coll = self._pick(1 if n_m < 9 else 2, new_share=0.3, start_day=int(t0 // DAY))
        ring_dev = [self.next_device + i for i in range(int(rng.integers(1, 4)))]
        self.next_device += len(ring_dev)
        dev = lambda a: ring_dev[rng.integers(len(ring_dev))] if rng.random() < 0.5 else self.dev1[a]  # noqa: E731
        slow, size = (8, 0.35) if stealth else (1, 1.0)
        inbound = {int(c): [] for c in coll}
        victims = self._victims(int(rng.integers(8, 41)))
        for v in victims:
            t = t0 + rng.uniform(0, dur) * DAY
            t = (t // DAY) * DAY + rng.choice(24, p=NIGHT_P if rng.random() < 0.4 else HOUR_P) * 3600 + rng.integers(0, 3600)
            amt = float(np.clip(rng.lognormal(6.7, 0.9), 100, 15_000)) * size
            m = int(rng.choice(mules))
            self._add(t, "transfer", "mobile", amt, src=v, dst=m, device=self.dev1[v])
            c = int(rng.choice(coll))
            # most mules forward within hours, but a good share sit on the money for a day or two
            t2 = t + (0.1 + rng.exponential(3.0 if rng.random() < 0.6 else 30.0)) * 3600 * slow
            fwd = amt * rng.uniform(0.60, 0.97)
            if rng.random() < 0.15 and n_m > 1:      # one extra hop through another mule
                m2 = int(rng.choice(mules[mules != m]))
                self._add(t2, "transfer", "mobile", fwd, src=m, dst=m2, device=dev(m))
                m, t2, fwd = m2, t2 + (0.1 + rng.exponential(2.0)) * 3600 * slow, fwd * rng.uniform(0.95, 0.99)
            pieces = [fwd] if rng.random() > 0.25 else [fwd * 0.55, fwd * 0.45]
            for i, p in enumerate(pieces):
                self._add(t2 + i * 900, "transfer", "mobile", p, src=m, dst=c, device=dev(m))
                inbound[c].append((t2 + i * 900, p))
        for c, items in inbound.items():
            items.sort()
            i = 0
            while i < len(items):
                batch = items[i:i + int(rng.integers(2, 7))]
                i += len(batch)
                t = batch[-1][0] + rng.uniform(1, 20) * 3600 * slow
                total = sum(p for _, p in batch) * rng.uniform(0.90, 0.99)
                if rng.random() < 0.6:
                    ext, ctry = self._risky_ext(0.7)
                    self._add(t, "wire_out", "web", total, src=c, ext=ext, country=ctry, device=dev(c))
                else:
                    k = int(rng.integers(2, 5))
                    for j in range(k):
                        self._add(t + j * rng.uniform(600, 7200), "cash_withdrawal", "atm", total / k, src=c)
        return {**{a: "mule" for a in mules}, **{a: "collector" for a in coll}}

    def _net_circular(self, stealth):
        """Funds travel round a closed loop of accounts and come back to where they started."""
        rng = self.rng
        k = int(rng.integers(3, 7))
        n_cycles = int(rng.integers(2, 7))
        dur = n_cycles * 6 + 6
        t0 = self._start(dur * (3 if stealth else 1))
        acc = self._pick(k, new_share=0.2, start_day=int(t0 // DAY))
        shared = self.next_device if rng.random() < 0.4 else None
        self.next_device += 1
        slow, size = (4, 0.35) if stealth else (1, 1.0)
        t = t0
        for c in range(n_cycles):
            amt = float(np.clip(rng.lognormal(8.8, 0.6), 2_000, 60_000)) * size
            order = np.roll(acc, -c % k)
            if rng.random() < 0.3:
                ext, ctry = self._risky_ext(0.4)
                self._add(t - rng.uniform(1, 12) * 3600, "wire_in", "wire", amt * 1.01, dst=order[0], ext=ext, country=ctry)
            for i in range(k):
                src, dst = order[i], order[(i + 1) % k]
                d = shared if shared is not None and rng.random() < 0.8 else self.dev1[src]
                self._add(t, "transfer", rng.choice(["mobile", "web"]), amt, src=src, dst=dst, device=d)
                amt *= rng.uniform(0.96, 0.995)
                t += rng.uniform(0.5, 20) * 3600 * slow
            t += rng.uniform(1, 6) * DAY * slow
        return {a: "cycle_member" for a in acc}

    def _net_structuring(self, stealth):
        """Smurfs deposit cash just under the threshold and consolidate it with a beneficiary."""
        rng = self.rng
        weeks = int(rng.integers(2, 7))
        t0 = self._start(weeks * 7 + 8)
        ben = int(self._pick(1, start_day=int(t0 // DAY))[0])
        smurfs = self._pick(int(rng.integers(2, 7)), new_share=0.3, start_day=int(t0 // DAY))
        lo, hi = (3_000, 6_500) if stealth else (7_500, 9_950)
        gap = 3.0 if stealth else 1.0
        arrivals = []
        for s_ in smurfs:
            t = t0 + rng.uniform(0, 4) * DAY
            pending, since = 0.0, 0
            for j in range(int(rng.integers(3, 9))):
                t += rng.uniform(1, 3) * DAY * gap
                dep_t = (t // DAY) * DAY + rng.integers(9, 17) * 3600 + rng.integers(0, 3600)
                amt = rng.uniform(lo, hi)
                self._add(dep_t, "cash_deposit", rng.choice(["branch", "atm"], p=[0.7, 0.3]), amt, dst=s_)
                pending, since = pending + amt, since + 1
                if since >= rng.integers(1, 3):
                    tt = dep_t + rng.uniform(2, 40) * 3600
                    self._add(tt, "transfer", "mobile", pending * rng.uniform(0.90, 0.99), src=s_, dst=ben,
                              device=self.dev1[s_])
                    arrivals.append((tt, pending))
                    pending, since = 0.0, 0
        arrivals.sort()
        i = 0
        while i < len(arrivals):
            batch = arrivals[i:i + int(rng.integers(2, 6))]
            i += len(batch)
            ext, ctry = self._risky_ext(0.5)
            self._add(batch[-1][0] + rng.uniform(4, 60) * 3600, "wire_out", "web",
                      sum(p for _, p in batch) * rng.uniform(0.92, 0.99), src=ben, ext=ext, country=ctry,
                      device=self.dev1[ben])
        return {ben: "beneficiary", **{a: "smurf" for a in smurfs}}

    def _net_layering(self, stealth):
        """A large inbound wire is passed along a chain of accounts within hours, then wired out."""
        rng = self.rng
        k = int(rng.integers(3, 8))
        n_ep = int(rng.integers(1, 5))
        t0 = self._start(n_ep * 12 + 6)
        acc = self._pick(k, new_share=0.4, start_day=int(t0 // DAY))
        shared = self.next_device if rng.random() < 0.3 else None
        self.next_device += 1
        slow, size = (10, 0.25) if stealth else (1, 1.0)
        t = t0
        for _ in range(n_ep):
            amt = float(np.clip(rng.lognormal(10.0, 0.6), 5_000, 250_000)) * size
            ext, ctry = self._risky_ext(0.5)
            self._add(t, "wire_in", "wire", amt, dst=acc[0], ext=ext, country=ctry)
            tt = t
            for i in range(k - 1):
                tt += rng.uniform(0.1, 6) * 3600 * slow
                amt *= rng.uniform(0.97, 0.995)
                d = shared if shared is not None and rng.random() < 0.8 else self.dev1[acc[i]]
                pieces = [amt] if rng.random() > 0.3 else [amt * 0.6, amt * 0.4]
                for j, p in enumerate(pieces):
                    self._add(tt + j * 600, "transfer", "web", p, src=acc[i], dst=acc[i + 1], device=d)
            ext, ctry = self._risky_ext(0.6)
            self._add(tt + rng.uniform(0.2, 8) * 3600 * slow, "wire_out", "web", amt * rng.uniform(0.97, 0.995),
                      src=acc[-1], ext=ext, country=ctry, device=self.dev1[acc[-1]])
            t += rng.uniform(3, 12) * DAY
        return {a: "layer" for a in acc}

    # ------------------------------------------------------------------ assemble
    def run(self):
        self.make_accounts()
        sus, nets, members = self.suspicious()     # first: it reserves accounts and fixes their opening dates
        tx = pd.concat([self.legit(), sus], ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
        tx.insert(0, "txn_id", np.arange(len(tx), dtype=np.int64))
        tx["ts"] = C.START + pd.to_timedelta(tx.ts, unit="s")
        tx["is_suspicious"] = tx.is_suspicious.astype("int8")
        for c in ("first_ts", "last_ts"):
            nets[c] = C.START + pd.to_timedelta(nets[c].clip(0, C.N_DAYS * DAY - 1), unit="s")
        accounts = pd.DataFrame({
            "account_id": np.arange(self.n), "customer_type": np.where(self.business, "business", "retail"),
            "segment": self.segment, "kyc_risk": self.kyc, "annual_income": np.round(self.income, 0),
            "open_date": C.START + pd.to_timedelta(self.open_day, unit="D"),
        })
        return accounts, tx, nets, members


def generate(scale: str = "full", out_dir: Path | None = None, seed: int = C.SEED) -> dict:
    out_dir = Path(out_dir or C.DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    accounts, tx, nets, members = Generator(C.SCALES[scale], seed).run()
    accounts.to_parquet(out_dir / "accounts.parquet", index=False)
    tx.to_parquet(out_dir / "transactions.parquet", index=False)
    nets.to_parquet(out_dir / "networks.parquet", index=False)
    members.to_parquet(out_dir / "network_members.parquet", index=False)
    return {"scale": scale, "seed": seed, "n_accounts": len(accounts), "n_transactions": len(tx),
            "n_suspicious_transactions": int(tx.is_suspicious.sum()), "n_networks": len(nets),
            "n_network_accounts": int(members.account_id.nunique()),
            "first_ts": str(tx.ts.min()), "last_ts": str(tx.ts.max())}


if __name__ == "__main__":
    import sys
    print(generate(sys.argv[1] if len(sys.argv) > 1 else "full"))
