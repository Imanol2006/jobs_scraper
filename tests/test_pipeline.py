"""Offline tests for the pure logic: location parsing, classification, fit scoring.

Nothing here touches the network — these are the invariants that used to break
silently when a keyword list was edited.

Run:  python -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import locations                                    # noqa: E402
from src.classify import (early_career_ok, function_ok,      # noqa: E402
                          max_years_required, track_of)
from src.fit import FitScorer                                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def cfg():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["classify"]


@pytest.fixture(scope="module")
def scorer():
    with open(os.path.join(ROOT, "profile.yaml"), encoding="utf-8") as f:
        return FitScorer(yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# locations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,metro", [
    ("San Francisco, CA", "SF Bay Area"),
    ("US, CA, Santa Clara", "SF Bay Area"),          # Workday's ordering
    ("Palo Alto", "SF Bay Area"),
    ("New York, New York, USA", "New York"),         # Amazon's normalized form
    ("Bellevue, Washington", "Seattle"),             # state name != DC's city
    ("Washington, DC", "DC Area"),
    ("Cambridge, MA", "Boston"),
    ("Hybrid - San Jose, CA", "SF Bay Area"),
    ("Greater Seattle Area", "Seattle"),
    ("Indianapolis, IN", "Other · IN"),              # "IN" survives tokenizing
    ("Portland, OR", "Portland"),                    # "OR" survives tokenizing
    ("LA", "Los Angeles"),                           # bare LA is the city
    ("Canoga Park, LA", "Los Angeles"),
    ("Baton Rouge, LA", "Other · LA"),               # a real Louisiana city
    ("Louisiana", "Other · LA"),                     # the spelled-out state
    ("London, UK", "London"),
    ("Bengaluru, India", "Bangalore"),
    ("Remote", "Remote"),
    # A remote role that still names a country keeps the country as its metro;
    # summarize() adds the "Remote" bucket on top, so it is findable under both.
    ("USA - Remote", "United States"),
    ("Remote in UK", "United Kingdom"),
])
def test_metro_parsing(raw, metro):
    assert locations.parse(raw).metro == metro


@pytest.mark.parametrize("raw,remote", [
    ("Remote", True), ("USA - Remote", True), ("Austin, TX; Remote", True),
    ("San Francisco, CA", False), ("Hybrid - San Jose, CA", False),
])
def test_remote_detection(raw, remote):
    assert locations.parse(raw).remote is remote


def test_summarize_merges_and_flags_remote():
    s = locations.summarize(["San Francisco, CA", "Remote - US", "London, UK"])
    # Every distinct bucket the role belongs to, so it is findable under each.
    assert s["metros"] == ["SF Bay Area", "United States", "London", "Remote"]
    assert s["remote"] is True and s["us"] is True
    assert "United Kingdom" in s["countries"]


def test_summarize_handles_empty():
    assert locations.summarize([])["metros"] == []
    assert locations.summarize(["", "  "])["metros"] == []


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("title,keep", [
    ("Machine Learning Engineer, New Grad", True),
    ("Software Engineer I", True),
    ("Data Scientist Intern", True),
    ("2026 University Graduate, Applied Scientist", True),
    ("Senior Machine Learning Engineer", False),
    ("Staff Research Scientist", False),
    ("Machine Learning Engineer III", False),
    ("Machine Learning Scientist 5", False),          # trailing numeric level
    ("Principal Data Scientist", False),
    ("Vice President, Engineering", False),           # 'resident' inside 'president'
    ("Technical Fellow", False),
])
def test_early_career_gate(cfg, title, keep):
    assert early_career_ok(title, "", "", cfg) is keep


def test_intern_keeps_roman_numeral(cfg):
    assert early_career_ok("Software Engineer Intern II", "", "intern", cfg) is True


def test_experience_bar_in_body_overrides_title(cfg):
    body = "Basic qualifications: 5+ years of professional experience in ML."
    assert early_career_ok("Machine Learning Engineer", "AI/ML", "new_grad", cfg) is True
    assert early_career_ok("Machine Learning Engineer", "AI/ML", "new_grad", cfg, body) is False


@pytest.mark.parametrize("text,years", [
    ("5+ years of experience", 5),
    ("minimum 3 years industry experience", 3),
    ("2-4 years of professional experience", 2),
    ("no requirement stated", 0),
    ("", 0),
])
def test_max_years_required(text, years):
    assert max_years_required(text) == years


@pytest.mark.parametrize("title,dept,ok", [
    ("Machine Learning Engineer", "Engineering", True),
    ("Enterprise Account Executive", "Sales", False),
    ("AI Tutor - Spanish", "Operations", False),
    ("Software Engineer", "Recruiting", False),
])
def test_function_gate(cfg, title, dept, ok):
    assert function_ok(title, dept, cfg) is ok


@pytest.mark.parametrize("title,track", [
    ("Machine Learning Engineer", "dsml"),
    ("Research Engineer, Interpretability", "dsml"),
    ("Member of Technical Staff", "dsml"),
    ("Data Engineer", "data_eng"),
    ("Software Engineer, Backend", "swe"),
    ("Quantitative Trader", "quant"),
    ("ASIC Design Engineer", "hardware"),
])
def test_track_of(cfg, title, track):
    assert track_of(title, "", cfg) == track


def test_title_beats_category_for_data_engineer(cfg):
    # A "Data/AI/ML" umbrella category must not drag Data Engineer into dsml.
    assert track_of("Data Engineer", "Data & AI/ML", cfg) == "data_eng"


# --------------------------------------------------------------------------- #
# fit scoring
# --------------------------------------------------------------------------- #
def _job(**kw):
    base = {"title": "", "company_name": "", "role_type": "new_grad", "category": "",
            "description": "", "employer_tier": "", "seasons": [], "years": [],
            "date_posted": 0, "active": True}
    return {**base, **kw}


def test_core_swe_role_outranks_offprofile_research_role(scorer):
    # This fork's profile.yaml targets broad software engineering first (Google
    # /Microsoft/Bloomberg-style roles), with AI/ML as a strong second interest
    # — not niche interpretability research, which role_penalties intentionally
    # docks. So a plain SWE role should outrank it, not the reverse.
    lab = scorer.score(_job(title="Research Engineer, Interpretability",
                            company_name="Anthropic", employer_tier="frontier_lab"))
    generic = scorer.score(_job(title="Software Engineer", company_name="Some Bank"))
    assert generic.score > lab.score
    assert any("Frontier" in r for r in lab.reasons)


def test_hardware_role_is_penalized(scorer):
    hw = scorer.score(_job(title="ASIC Verification Engineer, New Grad",
                           company_name="NVIDIA", employer_tier="big_tech"))
    ml = scorer.score(_job(title="Machine Learning Engineer, New Grad",
                           company_name="NVIDIA", employer_tier="big_tech"))
    assert ml.score > hw.score


def test_company_tier_inferred_from_name_for_aggregator_rows(scorer):
    # Aggregator feeds carry no tier; the name map has to supply it.
    assert scorer.tier_for("Google", "") == "big_tech"
    assert scorer.tier_for("OpenAI", "") == "frontier_lab"
    assert scorer.tier_for("Meta Platforms, Inc.", "") == "big_tech"
    assert scorer.tier_for("Some Local Credit Union", "") == ""


def test_declared_tier_wins_over_name_lookup(scorer):
    assert scorer.tier_for("Google", "ai_infra") == "ai_infra"


def test_phd_requirement_flags_and_penalizes(scorer):
    plain = scorer.score(_job(title="Applied Scientist", company_name="Amazon",
                              employer_tier="big_tech"))
    phd = scorer.score(_job(title="Applied Scientist", company_name="Amazon",
                            employer_tier="big_tech",
                            description="PhD in Computer Science required."))
    assert phd.score < plain.score
    assert "PhD mentioned" in phd.flags


def test_skills_overlap_raises_score(scorer):
    bare = scorer.score(_job(title="Machine Learning Engineer", company_name="Databricks",
                             employer_tier="ai_infra"))
    rich = scorer.score(_job(title="Machine Learning Engineer", company_name="Databricks",
                             employer_tier="ai_infra",
                             description="PyTorch, LangChain, RAG, FastAPI, Docker on AWS."))
    assert rich.score > bare.score
    assert any("stack:" in r for r in rich.reasons)


def test_intern_outranks_new_grad_all_else_equal(scorer):
    # This fork's profile.yaml sets stage.new_grad=0 on purpose: the candidate
    # is intern-only (sophomore, not graduating soon), so intern roles should
    # rank above new-grad roles, not below.
    ng = scorer.score(_job(title="Machine Learning Engineer", company_name="OpenAI",
                           employer_tier="frontier_lab", role_type="new_grad"))
    intern = scorer.score(_job(title="Machine Learning Engineer", company_name="OpenAI",
                               employer_tier="frontier_lab", role_type="intern"))
    assert intern.score > ng.score


def test_score_is_clamped_to_0_100(scorer):
    best = scorer.score(_job(
        title="Machine Learning Engineer, LLM", company_name="Anthropic",
        employer_tier="frontier_lab", role_type="new_grad",
        description="PyTorch LangChain RAG vLLM LoRA FastAPI Docker AWS GPU python sql",
        date_posted=2 ** 31 - 1))
    assert 0 <= best.score <= 100


def test_shortlist_caps_per_company(scorer):
    jobs = [_job(title=f"Machine Learning Engineer {i}", company_name="Anthropic",
                 employer_tier="frontier_lab") for i in range(10)]
    for j in jobs:
        j["fit"] = scorer.score(j).score
        j["key"] = j["title"]
    picked = scorer.shortlist(jobs)
    assert len(picked) == scorer.shortlist_cfg["max_per_company"]


def test_shortlist_reserves_slots_for_internships(scorer):
    """Full-time always outscores an internship, so without a reservation the
    intern path vanishes from the picks entirely."""
    jobs = []
    for i in range(200):
        j = _job(title=f"Machine Learning Engineer {i}", company_name=f"Lab{i}",
                 employer_tier="frontier_lab", role_type="new_grad")
        j.update(fit=95, key=f"ng{i}")
        jobs.append(j)
    for i in range(30):
        j = _job(title=f"ML Intern {i}", company_name=f"Lab{i}",
                 employer_tier="frontier_lab", role_type="intern")
        j.update(fit=80, key=f"in{i}")
        jobs.append(j)
    picked = scorer.shortlist(jobs)
    n_intern = sum(1 for j in picked if j["role_type"] == "intern")
    assert len(picked) == scorer.shortlist_cfg["max_size"]
    assert n_intern == scorer.shortlist_cfg["reserve_intern"]


def test_shortlist_backfills_when_few_interns_qualify(scorer):
    jobs = []
    for i in range(200):
        j = _job(title=f"Machine Learning Engineer {i}", company_name=f"Lab{i}",
                 employer_tier="frontier_lab", role_type="new_grad")
        j.update(fit=95, key=f"ng{i}")
        jobs.append(j)
    picked = scorer.shortlist(jobs)
    # No internships exist, so the reserved slots go back to full-time roles.
    assert len(picked) == scorer.shortlist_cfg["max_size"]


def test_shortlist_caps_research_titles(scorer):
    """Frontier labs title nearly everything "Research Engineer"; unchecked they
    took 40% of the list."""
    jobs = []
    for i in range(200):
        j = _job(title=f"Research Engineer, Team {i}", company_name=f"Lab{i}",
                 employer_tier="frontier_lab")
        j.update(fit=95, key=f"res{i}")
        jobs.append(j)
    for i in range(200):
        j = _job(title=f"Machine Learning Engineer {i}", company_name=f"Co{i}",
                 employer_tier="ai_infra")
        j.update(fit=90, key=f"mle{i}")
        jobs.append(j)
    picked = scorer.shortlist(jobs)
    n_res = sum(1 for j in picked if "research" in j["title"].lower())
    assert n_res == scorer.shortlist_cfg["max_research"]
    assert len(picked) == scorer.shortlist_cfg["max_size"]


def test_applied_role_outranks_research_role_at_same_employer(scorer):
    research = scorer.score(_job(title="Research Engineer, Post-Training",
                                 company_name="Anthropic", employer_tier="frontier_lab"))
    applied = scorer.score(_job(title="Machine Learning Engineer, Product",
                                company_name="Anthropic", employer_tier="frontier_lab"))
    assert applied.score > research.score


def test_shortlist_excludes_inactive_and_low_scores(scorer):
    good = _job(title="ML Engineer", company_name="OpenAI", employer_tier="frontier_lab")
    good.update(fit=95, key="a")
    stale = _job(title="ML Engineer", company_name="Cohere", employer_tier="frontier_lab",
                 active=False)
    stale.update(fit=95, key="b")
    weak = _job(title="Barista", company_name="Cafe")
    weak.update(fit=5, key="c")
    picked = scorer.shortlist([good, stale, weak])
    assert [j["key"] for j in picked] == ["a"]
