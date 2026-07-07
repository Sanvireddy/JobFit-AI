"""Unit tests for the pure matching logic (no DB, no model, no network)."""

from app.matching.matcher import filter_jobs_by_metadata, metadata_from_row


def job(job_id="1"):
    return {"job_id": job_id, "title": "ML Engineer", "company": "Acme"}


class TestFilterJobsByMetadata:
    def test_drops_jobs_requiring_more_experience(self):
        metadata = {"1": {"min_experience_years": 5}}
        assert filter_jobs_by_metadata([job()], metadata, candidate_experience_years=3) == []

    def test_keeps_jobs_within_experience(self):
        metadata = {"1": {"min_experience_years": 2}}
        assert len(filter_jobs_by_metadata([job()], metadata, candidate_experience_years=3)) == 1

    def test_skips_experience_check_when_candidate_years_unknown(self):
        metadata = {"1": {"min_experience_years": 10}}
        assert len(filter_jobs_by_metadata([job()], metadata, candidate_experience_years=None)) == 1

    def test_tolerates_non_numeric_experience_values(self):
        metadata = {"1": {"min_experience_years": "senior"}}
        assert len(filter_jobs_by_metadata([job()], metadata, candidate_experience_years=1)) == 1

    def test_drops_jobs_requiring_advanced_degree(self):
        metadata = {"1": {"requires_advanced_degree": True}}
        assert filter_jobs_by_metadata([job()], metadata) == []

    def test_drops_jobs_requiring_non_english_language(self):
        metadata = {"1": {"requires_only_english": False}}
        assert filter_jobs_by_metadata([job()], metadata) == []

    def test_keeps_jobs_with_unknown_language_requirement(self):
        metadata = {"1": {"requires_only_english": None}}
        assert len(filter_jobs_by_metadata([job()], metadata)) == 1

    def test_keeps_jobs_with_no_metadata_row(self):
        assert len(filter_jobs_by_metadata([job()], {})) == 1

    def test_drops_jobs_missing_a_must_have_skill(self):
        jobs = [dict(job(), description="We use Python and SQL daily.")]
        assert filter_jobs_by_metadata(
            jobs, {}, must_have_skills=["python", "aws"]
        ) == []

    def test_keeps_jobs_mentioning_all_must_have_skills(self):
        jobs = [dict(job(), description="We use Python and SQL daily.")]
        assert len(
            filter_jobs_by_metadata(jobs, {}, must_have_skills=["Python", "sql"])
        ) == 1

    def test_drops_jobs_outside_preferred_locations(self):
        jobs = [dict(job(), location="Bengaluru, India", country="India")]
        assert filter_jobs_by_metadata(
            jobs, {}, preferred_locations=["Germany"]
        ) == []

    def test_keeps_jobs_in_preferred_locations(self):
        jobs = [dict(job(), location="Berlin, Germany", country="Germany")]
        assert len(
            filter_jobs_by_metadata(jobs, {}, preferred_locations=["germany"])
        ) == 1

    def test_remote_jobs_bypass_location_preference(self):
        jobs = [dict(job(), location="Bengaluru, India", country="India")]
        metadata = {"1": {"work_mode": "remote"}}
        assert len(
            filter_jobs_by_metadata(jobs, metadata, preferred_locations=["Germany"])
        ) == 1

    def test_relocation_openness_bypasses_location_preference(self):
        jobs = [dict(job(), location="Bengaluru, India", country="India")]
        assert len(filter_jobs_by_metadata(
            jobs, {}, preferred_locations=["Germany"], open_to_relocation=True
        )) == 1

    def test_drops_jobs_explicitly_refusing_sponsorship_when_needed(self):
        metadata = {"1": {"visa_sponsorship_available": False}}
        assert filter_jobs_by_metadata(
            [job()], metadata, requires_visa_sponsorship=True
        ) == []

    def test_keeps_jobs_with_unknown_sponsorship_when_needed(self):
        metadata = {"1": {"visa_sponsorship_available": None}}
        assert len(filter_jobs_by_metadata(
            [job()], metadata, requires_visa_sponsorship=True
        )) == 1


class TestMetadataFromRow:
    def test_none_or_empty_row_returns_none(self):
        assert metadata_from_row(None) is None
        assert metadata_from_row({}) is None

    def test_row_with_only_null_fields_returns_none(self):
        row = {
            "min_experience_years": None,
            "experience_requirement_text": None,
            "requires_advanced_degree": None,
            "education_requirement_text": None,
            "relocation_evidence": None,
        }
        assert metadata_from_row(row) is None

    def test_full_row_roundtrips(self):
        row = {
            "min_experience_years": 3,
            "experience_requirement_text": "3+ years of experience",
            "requires_advanced_degree": False,
            "education_requirement_text": "Bachelor's degree required",
            "visa_sponsorship_available": True,
            "relocation_assistance_provided": None,
            "work_mode": "hybrid",
            "relocation_evidence": "Visa sponsorship available",
        }
        metadata = metadata_from_row(row)
        assert metadata.experience_requirement.min_years_experience == 3
        assert metadata.higher_education_requirement.is_masters_or_phd_required is False
        assert metadata.relocation_requirement.work_mode == "hybrid"
        assert metadata.relocation_requirement.visa_sponsorship_available is True

    def test_partial_row_builds_only_available_submodels(self):
        row = {
            "min_experience_years": 2,
            "experience_requirement_text": "2 years",
            "relocation_evidence": None,
        }
        metadata = metadata_from_row(row)
        assert metadata.experience_requirement.min_years_experience == 2
        assert metadata.relocation_requirement is None
