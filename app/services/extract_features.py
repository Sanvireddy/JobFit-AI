
# from google import genai
from app.ingestion.helpers import compute_is_only_english_required
from app.ingestion.metata_extractor import extract_metadata
# from langdetect import detect
# from deep_translator import GoogleTranslator
from app.db.insert_data import get_data_from_scraped_jobs, insert_jobs_metadata, mark_jobs_processed


prompt = """
You are an information extraction system.

Analyze the job description and extract only explicitly stated information about:

1. Minimum years of experience required
2. Language requirements mentioned in the job description
3. Whether a Master's or PhD degree is explicitly required
4. Relocation / international candidate / location policy

##YEARS OF EXPERIENCE

Extract the minimum number of years of experience required.

Examples:
- "Minimum 3 years of experience" → 3
- "5+ years experience" → 5
- "2–4 years experience" → 2
- If not clearly stated → return null

##LANGUAGE REQUIREMENTS

For each language mentioned, classify it as:

- "required" → if the language is clearly mandatory
- "good_to_have" → if the language is optional, preferred, or a plus
- "mentioned" → if the language is referenced but not clearly required

Examples:

- "Fluent Dutch required" → required
- "German is a plus" → good_to_have
- "You will work with German clients" → mentioned

Extract language proficiency level if explicitly stated (e.g. A1, B2, C1).

## MASTER'S OR PHD REQUIREMENT

Determine whether the job description explicitly requires a Master's degree or PhD.

Return:
- true → if Master's or PhD is explicitly required
- false → if Bachelor's is sufficient or if Master's/PhD are optional
- null → if the description does not clearly specify

Examples:

- "Master's degree required" → true
- "PhD preferred" → false
- "Bachelor's or Master's degree" → false
- "Bachelor's degree required" → false
- If not mentioned → null

-------------------------------------

## RELOCATION / LOCATION POLICY

Return exactly one value for "relocation_policy" from this list:

- "relocation_provided"
- "visa_sponsorship_provided"
- "international_candidates_allowed"
- "must_be_in_location"
- "remote_outside_country_allowed"
- "no_relocation_or_sponsorship"
- "unknown"

Rules:

Use "relocation_provided" only if relocation support is explicitly mentioned.

Use "visa_sponsorship_provided" only if visa sponsorship is explicitly mentioned.

Use "international_candidates_allowed" only if the description explicitly says international applicants are welcome.

Use "must_be_in_location" only if the description explicitly requires candidates to already be in a specific location or to have work authorization there.

Use "remote_outside_country_allowed" only if the role explicitly allows remote work from another country.

Use "no_relocation_or_sponsorship" only if the description explicitly states no relocation or no visa sponsorship.

Use "unknown" if none of the above is clearly stated.


## IMPORTANT RULES

- Only extract information explicitly stated in the job description.
- Do NOT infer facts that are not clearly mentioned.
- Do NOT assume relocation or sponsorship from job location alone.
- If uncertain, return null or "unknown".
- Include short evidence snippets copied exactly from the job description.

Do NOT wrap the output in markdown code blocks.

Return strictly valid JSON using this schema:

{
  "minimum_years_experience": integer | null,
  "minimum_years_experience_evidence": "string" | null, 
  "is_masters_or_phd_required": true | false | null,
  "is_masters_or_phd_required_evidence": "string" | null,
  "languages": [
    {
      "language": "string",
      "level": "string" | null,
      "requirement": "required | good_to_have | mentioned",
      "evidence": "string"
    }
  ],
  "relocation_policy": "relocation_provided | visa_sponsorship_provided | international_candidates_allowed | must_be_in_location | remote_outside_country_allowed | no_relocation_or_sponsorship | unknown",
  "relocation_evidence": "string" | null
}

## JOB DESCRIPTION:

"""
def process_all_job_ids():
    batch_size = 5
    failed_jobs = []
    processed_count = 0

    def flush_batch(job_ids_batch):
        if not job_ids_batch:
            return 0

        try:
            insert_jobs_metadata(job_ids_batch)
            job_ids = [item[0] for item in job_ids_batch]
            mark_jobs_processed(job_ids)
            print(f"Inserted and marked {len(job_ids_batch)} job IDs as processed")
            return len(job_ids_batch)
        except Exception as exc:
            failed_jobs.extend(
                [(item[0], str(exc)) for item in job_ids_batch]
            )
            print(f"Batch insert failed for {len(job_ids_batch)} jobs: {exc}")
            return 0

    while True:
        response_list = get_data_from_scraped_jobs("job_id, description")
        if not response_list:
            break

        job_ids_batch = []
        for response in response_list:
            try:
                meta_data = extract_metadata(response[1])

                is_only_english_required, english_evidence = (
                    compute_is_only_english_required(
                        meta_data.language_requirements
                    )
                )

                experience_requirement = meta_data.experience_requirement
                higher_education_requirement = meta_data.higher_education_requirement
                relocation_requirement = meta_data.relocation_requirement

                job_ids_batch.append(
                    (
                        response[0],
                        experience_requirement.min_years_experience
                        if experience_requirement else None,
                        experience_requirement.experience_requirement_evidence
                        if experience_requirement else None,
                        is_only_english_required,
                        english_evidence,
                        higher_education_requirement.is_masters_or_phd_required
                        if higher_education_requirement else None,
                        higher_education_requirement.education_requirement_evidence
                        if higher_education_requirement else None,
                        relocation_requirement.visa_sponsorship_available
                        if relocation_requirement else None,
                        relocation_requirement.relocation_assistance_provided
                        if relocation_requirement else None,
                        relocation_requirement.work_mode
                        if relocation_requirement else None,
                        relocation_requirement.relocation_evidence
                        if relocation_requirement else None,
                    )
                )
            except Exception as exc:
                failed_jobs.append((response[0], str(exc)))

            if len(job_ids_batch) >= batch_size:
                processed_count += flush_batch(job_ids_batch)
                job_ids_batch = []

        if job_ids_batch:
            processed_count += flush_batch(job_ids_batch)

    if failed_jobs:
        print("Failed jobs:")
        for job_id, error in failed_jobs:
            print(f"- {job_id}: {error}")
    else:
        print("No failed jobs.")

    print(f"Finished processing all job IDs. Total processed: {processed_count}.")

process_all_job_ids()

# def extract_english_required_field(languages):
#   for language in languages:
#     if language['language']!='English':
#       if language['requirement']=='required':
#         return 0
#   return 1
  
# def process_language_requirement_field(languages):
#   final_requirement = []
#   for language in languages:
#     final_requirement.append(f"{language['language']} with level {language['level']} and {language['requirement']}")
#   return ", ".join(final_requirement)
    
      
  
# def update_other_fields_from_prompt():
#     results = update_scraped_jobs_fields_from_prompt_response()
#     for result in results:
#       job_id, job_desc = result
#       print(job_id, job_desc)

# # update_extracted_prompt_attributes(
# # generate_response_from_llm(prompt,"4383542485","""
# #                           Aufgaben, die mich erwarten aktives Mitwirken bei der Entwicklung und Optimierung unserer WarenbedarfsplanungUnterstützen bei quantitativen sowie qualitativen Analysen unserer Produkte und WarengruppenErmitteln, Validieren und Optimieren von Prognoseparametern unter Berücksichtigung der Bedarfseinflussfaktorenselbstständiges Ausarbeiten und Simulieren von Szenarien zur Verbesserung unserer Beschaffungsstrategieenges Zusammenarbeiten und kontinuierliches Abstimmen von Ergebnissen mit internen Abteilungen Qualifikationen, die ich mitbringe laufendes oder abgeschlossenes Studium mit Schwerpunkt Mathematik, Statistik, Informatik, Wirtschaftswissenschaften oder Supply Chain Management Berufserfahrung im Umgang mit Datenverarbeitung fundierte Kenntnisse in Excel, Erfahrungen mit SAP, RStudio sowie Programmierkenntnisse von Vorteilsehr gute Deutsch- und gute Englischkenntnissepotenzielle Bereitschaft zum Dienstbeginn um 04:00 Uhr morgensausgeprägtes Interesse für DatenanalyseBereitschaft sich in neue Bereiche einzuarbeiten und innovative Lösungen zu entwickelnanalytische, selbstständige und lösungsorientierte Arbeitsweise Angebote, die mich überzeugen umfangreiche Einarbeitung und individuelles Onboarding flexible Gleitzeitmodelle sowie bis zu 50% Home-Office/Remote Work top ausgestatteter ergonomischer Arbeitsplatz sowie eine moderne technische Ausstattung kostenlose Verpflegung in Form von täglich frischem Obst und Gemüse, Kaffee sowie TeeAus- und Weiterbildungsmöglichkeiten, wie interaktive E-Learnings oder digitale Lernplattformen im Rahmen der HOFER AKADEMIEsicherer und verlässlicher ArbeitgeberDU-Kultur im ganzen UnternehmenMöglichkeit von mehrmonatigen Sabbaticalsvergünstigte Tarife bei KrankenzusatzversicherungenLeasingprogramm für Fahrräder und E-Bikes Unterstützung bei Pflegefällen mit unserer Pflegeplattform Entgelt attraktives Brutto-Monatseinstiegsgehalt ab € 3.730,- für 38,5 Stunden/Woche, abhängig von Qualifikation und Berufserfahrung bis € 4.659,- in der Endstufe Arbeitsort Hofer Straße 1, 4642 Sattledt Arbeitsbeginn ab sofort Online bewerben Jetzt online bewerben und Lebenslauf sowie sämtliche relevante Zeugnisse beifügen.
# #                           """
# # ))
# import re
# from deep_translator import GoogleTranslator


# def translate_long_text(text: str, target_lang: str = "en", source_lang: str = "auto", max_chars: int = 4500) -> str:
#     """
#     Translate long text safely by splitting it into smaller chunks.

#     Args:
#         text: Input text to translate
#         target_lang: Target language code (default: 'en')
#         source_lang: Source language code (default: 'auto')
#         max_chars: Max chars per chunk, kept below provider limit

#     Returns:
#         Translated text as a single string
#     """
#     if not text or not text.strip():
#         return text

#     def split_into_sentences(text: str):
#         # Basic sentence split
#         sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
#         return [s.strip() for s in sentences if s.strip()]

#     def make_chunks(sentences, max_chars):
#         chunks = []
#         current_chunk = ""

#         for sentence in sentences:
#             # If one sentence itself is too long, split it further
#             if len(sentence) > max_chars:
#                 if current_chunk:
#                     chunks.append(current_chunk.strip())
#                     current_chunk = ""

#                 for i in range(0, len(sentence), max_chars):
#                     chunks.append(sentence[i:i + max_chars].strip())
#                 continue

#             # Add sentence to current chunk if it fits
#             if len(current_chunk) + len(sentence) + 1 <= max_chars:
#                 current_chunk += " " + sentence if current_chunk else sentence
#             else:
#                 chunks.append(current_chunk.strip())
#                 current_chunk = sentence

#         if current_chunk:
#             chunks.append(current_chunk.strip())

#         return chunks

#     sentences = split_into_sentences(text)
#     chunks = make_chunks(sentences, max_chars)

#     translator = GoogleTranslator(source=source_lang, target=target_lang)
#     translated_chunks = []

#     for chunk in chunks:
#         translated_chunk = translator.translate(chunk)
#         translated_chunks.append(translated_chunk)

#     return " ".join(translated_chunks)
  