from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LanguageRequirement(BaseModel):
    language: str = Field(
        description="Language mentioned in the job description, such as English, German, or French."
    )

    level: Optional[str] = Field(
        default=None,
        description="Required proficiency level for the language, such as fluent, native, B2, C1, or conversational."
    )

    requirement: Literal[
        "required",
        "good_to_have",
        "not_required",
        "mentioned"
    ] = Field(
        default="required",
        description=(
            "Indicates whether the language is mandatory ('required'), optional/preferred "
            "('good_to_have'), explicitly not required ('not_required'), or simply referenced "
            "without being a requirement ('mentioned'). The extraction prompt instructs the "
            "model to emit 'mentioned', so it must be an accepted value here."
        )
    )

    evidence: str = Field(
        description="Exact sentence or phrase from the job description supporting the extracted language requirement."
    )


class ExperienceRequirement(BaseModel):
    min_years_experience: Optional[int] = Field(
        default=None,
        description="Minimum number of years of professional experience required for the role."
    )

    experience_requirement_evidence: str = Field(
        description="Exact sentence or phrase from the job description supporting the extracted experience requirement."
    )

class HigherEducationRequirement(BaseModel):

    is_masters_or_phd_required: Optional[bool] = Field(
        default=None,
        description=(
            "True if the job description explicitly requires a Master's degree or PhD. "
            "False if Master's/PhD are optional, preferred, or if Bachelor's is sufficient. "
            "Null if unclear or not mentioned."
        )
    )

    education_requirement_evidence: Optional[str] = Field(
        default=None,
        description=(
            "Exact sentence or phrase from the job description "
            "supporting the extracted education requirement."
        )
    )


class RelocationRequirement(BaseModel):
    visa_sponsorship_available: Optional[bool] = Field(
        default=None,
        description="Whether the company mentions providing visa sponsorship for the role."
    )

    relocation_assistance_provided: Optional[bool] = Field(
        default=None,
        description="Whether the company mentions providing relocation support or relocation assistance."
    )

    work_mode: Literal[
        "remote",
        "hybrid",
        "onsite",
        "unknown"
    ] = Field(
        default="unknown",
        description="Work arrangement mentioned in the job description."
    )

    relocation_evidence: str = Field(
        description="Exact sentence or phrase from the job description supporting the extracted relocation or work mode information."
    )


class JobMetadata(BaseModel):
    language_requirements: Optional[List[LanguageRequirement]] = Field(
        default=None,
        description="List of language requirements mentioned in the job description."
    )

    experience_requirement: Optional[ExperienceRequirement] = Field(
        default=None,
        description="Experience requirement extracted from the job description."
    )

    higher_education_requirement: Optional[HigherEducationRequirement] = Field(
        default=None,
        description="Higher education requirement extracted from the job description."
    )

    relocation_requirement: Optional[RelocationRequirement] = Field(
        default=None,
        description="Relocation, visa sponsorship, and work mode details extracted from the job description."
    )

