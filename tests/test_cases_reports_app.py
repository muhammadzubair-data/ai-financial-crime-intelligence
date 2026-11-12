import glob
import json
from pathlib import Path

import pandas as pd
import pytest

from fincrime.cases.store import CaseStore
from fincrime.reports.case_report import build_case_report, render_html, render_markdown, to_json

ROOT = Path(__file__).resolve().parents[1]


def test_case_lifecycle_and_audit_trail(tmp_path):
    store = CaseStore(tmp_path / "cases.db")
    cid = store.open_case(42, "ALT-00001", "High", 0.6, actor="amy")
    assert store.open_case(42) == cid, "one case per account"
    store.update(cid, actor="amy", status="In review", assigned_to="amy")
    store.add_note(cid, "amy", "Checked counterparties.")
    store.update(cid, actor="bob", status="Closed", disposition="False positive")
    case = store.by_account(42)
    assert (case["status"], case["disposition"], case["assigned_to"]) == ("Closed", "False positive", "amy")
    assert len(store.notes(cid)) == 1
    assert list(store.history(cid).action) == ["opened", "status changed", "assigned_to changed", "note added",
                                               "status changed", "disposition changed"]
    with pytest.raises(ValueError):
        store.update(cid, status="Deleted")
    with pytest.raises(ValueError):
        store.add_note(cid, "amy", "   ")
    assert len(CaseStore(tmp_path / "cases.db").list()) == 1, "cases persist on disk"


def test_case_report_renders_in_all_formats(run, artifacts):
    data = {**artifacts, "summary": run["summary"]}
    account_id = int(artifacts["alerts"].account_id.iloc[0])
    case = {"case_id": 1, "status": "Open", "disposition": "Undecided", "assigned_to": "", "opened_at": "x", "updated_at": "x"}
    notes = pd.DataFrame({"created_at": ["t"], "author": ["amy"], "body": ["<b>note</b> | pipe"]})
    report = build_case_report(account_id, data, case, notes, "amy")
    titles = [s[0] for s in report["sections"]]
    for needed in ("Subject", "Risk summary", "Why this account was flagged", "Network context", "Investigation record"):
        assert needed in titles
    md, html_, js = render_markdown(report), render_html(report), to_json(report)
    assert "SYNTHETIC DATA" in md and "SYNTHETIC DATA" in html_
    assert "&lt;b&gt;note&lt;/b&gt;" in html_, "user text is escaped in HTML"
    assert json.loads(js)["account_id"] == account_id


@pytest.mark.parametrize("script", ["app/Home.py"] + sorted(glob.glob("app/pages/*.py", root_dir=ROOT)))
def test_app_page_runs_without_error(run, script):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / script), default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_workbench_can_open_and_decide_a_case(run):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app/pages/3_Investigation_Workbench.py"), default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception
    at.selectbox[1].set_value("Escalated")
    at.text_area[0].input("Looks like a mule account.")
    next(b for b in at.button if b.label == "Save").click().run()
    assert not at.exception
    from fincrime import config as C
    cases = CaseStore(C.CASE_DB).list()
    assert len(cases) == 1 and cases.status.iloc[0] == "Escalated"
