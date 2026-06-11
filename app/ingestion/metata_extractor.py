from app.schemas.job_metadata import JobMetadata
import ollama
import json

def build_prompt(job_description):
    with open('app/llm/prompts/extraction_prompt.txt', 'r') as file:
        prompt_template = file.read()
    json_schema = JobMetadata.model_json_schema()
    prompt_template = prompt_template.replace("{output_json}", json.dumps(json_schema, indent=2))
    prompt = prompt_template.replace("{job_description}", job_description)
    return prompt
build_prompt("Hey")

def extract_metadata(job_description, MAX_RETRIES=3):
    last_error = None
    for attempt in range(MAX_RETRIES):
        prompt = build_prompt(job_description)
        if last_error:
            prompt += (
                f"\n\n## PREVIOUS ATTEMPT FAILED\n"
                f"Your previous response failed Pydantic validation with this error:\n"
                f"{last_error}\n"
                f"Return corrected JSON that fixes this specific issue. "
                f"Do NOT repeat the same mistake."
            )
        response = ollama.chat(model="qwen2.5:7b", messages=[{"role": "user", "content": prompt}])
        content = response['message']['content']
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()
        try:
            metadata_dict = json.loads(content)
            # Validate with Pydantic
            validated_output = JobMetadata.model_validate(
                metadata_dict
            )
            return validated_output
        except Exception as e:
            last_error = str(e)
            print(
                f"Attempt {attempt + 1} failed: {e}"
            )

    raise Exception(
        "Failed to extract valid metadata after retries."
    )

