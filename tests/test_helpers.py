from app.ingestion.helpers import clean_job_desc, compute_is_only_english_required
from app.schemas.job_metadata import LanguageRequirement


def lang(language, requirement="required", evidence="mentioned in posting"):
    return LanguageRequirement(
        language=language, requirement=requirement, evidence=evidence
    )


class TestComputeIsOnlyEnglishRequired:
    def test_none_when_no_languages(self):
        assert compute_is_only_english_required(None) == (None, None)

    def test_none_when_nothing_is_required(self):
        langs = [lang("German", requirement="good_to_have")]
        assert compute_is_only_english_required(langs) == (None, None)

    def test_true_when_only_english_required(self):
        result, evidence = compute_is_only_english_required([lang("English")])
        assert result is True
        assert evidence == "mentioned in posting"

    def test_false_when_non_english_required(self):
        result, _ = compute_is_only_english_required(
            [lang("English"), lang("German")]
        )
        assert result is False

    def test_case_insensitive_language_names(self):
        result, _ = compute_is_only_english_required([lang("ENGLISH")])
        assert result is True


class TestCleanJobDesc:
    def test_collapses_whitespace(self):
        assert clean_job_desc("  a   b \n\n c ") == "a b c"
