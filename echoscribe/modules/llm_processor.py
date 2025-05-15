# echoscribe/modules/llm_processor.py

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
import shutil # For backing up the cumulative summary

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import google.auth.exceptions
from google.api_core import exceptions as google_api_exceptions

log = logging.getLogger(__name__)

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = MODULE_DIR / "prompts"

DEFAULT_NARRATIVE_PROMPT_FILENAME = "default_narrative_prompt.txt"
DEFAULT_SUMMARY_PROMPT_FILENAME = "default_summary_prompt.txt"
DEFAULT_CUMULATIVE_SUMMARY_PROMPT_FILENAME = "default_cumulative_summary_prompt.txt"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM processing, including cumulative summary logic."""
    api_provider: str = "gemini"
    gemini_api_key_env_var: str = "GEMINI_API_KEY"
    model_name_summary: str = "gemini-1.5-flash-latest"
    model_name_narrative: str = "gemini-1.5-pro-latest"

    language: str = "en"
    narrative_prompt_filename: str = DEFAULT_NARRATIVE_PROMPT_FILENAME
    summary_prompt_filename: str = DEFAULT_SUMMARY_PROMPT_FILENAME
    cumulative_summary_prompt_filename: str = DEFAULT_CUMULATIVE_SUMMARY_PROMPT_FILENAME

    output_summary_filename: str = "llm_current_session_summary.txt"
    output_narrative_filename: str = "llm_current_session_narrative.txt"
    output_cumulative_meta_summary_filename: str = "llm_campaign_cumulative_summary.txt"

    max_output_tokens_summary: int = 1000
    max_output_tokens_narrative: int = 4000
    max_output_tokens_cumulative: int = 2000
    temperature: float = 0.7
    safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = field(default_factory=lambda: {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    })

    enable_summary: bool = True
    enable_narrative: bool = True
    enable_cumulative_summary: bool = True

    # Corrected fallback prompt placeholders
    fallback_narrative_prompt: str = (
        "Narrate this session script: {current_session_script}\n"
        "Context from previous sessions: {cumulative_previous_sessions_summary}"
    )
    fallback_summary_prompt: str = (
        "Summarize this session script: {current_session_script}\n"
        "Context from previous sessions: {cumulative_previous_sessions_summary}"
    )
    fallback_cumulative_summary_prompt: str = (
        "Update campaign summary.\n"
        "Previous Overall Campaign Summary:\n{previous_cumulative_summary}\n\n"
        "Summary of the Latest Session to Integrate:\n{current_session_concise_summary}\n\n"
        "Produce the UPDATED AND INTEGRATED campaign summary below:"
    )

    initial_campaign_context: str = "This is the very beginning of the campaign. No previous events have been recorded."

    def __post_init__(self):
        # Basic configuration validation
        if not self.model_name_summary:
            raise ValueError("LLMConfig: model_name_summary cannot be empty.")
        if not self.model_name_narrative:
            raise ValueError("LLMConfig: model_name_narrative cannot be empty.")
        if self.max_output_tokens_summary <= 0:
            raise ValueError("LLMConfig: max_output_tokens_summary must be positive.")
        if self.max_output_tokens_narrative <= 0:
            raise ValueError("LLMConfig: max_output_tokens_narrative must be positive.")
        if self.max_output_tokens_cumulative <= 0:
            raise ValueError("LLMConfig: max_output_tokens_cumulative must be positive.")
        if not (0.0 <= self.temperature <= 2.0): # Gemini typical range is 0-1, some models allow up to 2
            raise ValueError("LLMConfig: temperature must be between 0.0 and 2.0 (typically 0.0-1.0).")
        if not self.language:
            raise ValueError("LLMConfig: language cannot be empty.")


DEFAULT_LLM_CONFIG = LLMConfig()


def _load_api_key(env_var_name: str) -> Optional[str]:
    api_key = os.getenv(env_var_name)
    if not api_key:
        log.warning(f"API key environment variable '{env_var_name}' not found or is empty.")
    return api_key

def _load_prompt_template(
    base_prompts_dir: Path, language: str, prompt_filename: str, fallback_prompt: str
) -> str:
    prompt_file_path = base_prompts_dir / language / prompt_filename
    try:
        if prompt_file_path.is_file():
            log.info(f"Loading prompt from: {prompt_file_path}")
            return prompt_file_path.read_text(encoding="utf-8")
        else:
            log.warning(f"Prompt file not found: {prompt_file_path}. Using fallback prompt.")
            return fallback_prompt
    except FileNotFoundError: # More specific for this case
        log.warning(f"Prompt file specifically not found (FileNotFoundError): {prompt_file_path}. Using fallback.")
        return fallback_prompt
    except IOError as e:
        log.error(f"Error reading prompt file {prompt_file_path}: {e}. Using fallback prompt.")
        return fallback_prompt
    except Exception as e:
        log.exception(f"Unexpected error loading prompt file {prompt_file_path}: {e}. Using fallback.")
        return fallback_prompt


def _call_gemini_api(
    api_key: str, model_name: str, prompt_content: str, max_tokens: int,
    temperature: float, safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]]
) -> str:
    """
    Calls the Google Gemini API to generate content. (Docstring from before, still relevant)
    Brief explanation of common API errors caught:
    - BlockedPromptException: Prompt violated safety policies.
    - RefreshError/GoogleAuthError: API key or authentication problems.
    - PermissionDenied: Key lacks permission for the API/project.
    - ResourceExhausted: Rate limit or quota exceeded.
    - InvalidArgument: Bad request (e.g., wrong model name, bad params).
    - NotFound: Requested resource (e.g., model) doesn't exist.
    - Unavailable/InternalServerError/DeadlineExceeded: Temporary server-side issues or timeouts.
    """
    log.info(f"Calling Gemini API with model: {model_name}")
    log.debug(f"Prompt content (first 200 chars): {prompt_content[:200]}...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=max_tokens, temperature=temperature
        )
        response = model.generate_content(
            prompt_content, generation_config=generation_config, safety_settings=safety_settings
        )
        if not response.candidates:
            block_reason_name = "Unknown (no candidates returned)"
            prompt_feedback = getattr(response, 'prompt_feedback', None)
            if prompt_feedback and getattr(prompt_feedback, 'block_reason', None):
                block_reason_name = prompt_feedback.block_reason.name
            log.error(f"Gemini API model '{model_name}' call: no candidates. Blocked? Reason: {block_reason_name}")
            if prompt_feedback:
                for rating in getattr(prompt_feedback, 'safety_ratings', []):
                    log.error(f"Safety Rating Detail: Category '{rating.category.name}', Probability '{rating.probability.name}'")
            raise RuntimeError(f"Gemini API '{model_name}' blocked/failed to produce candidates. Reason: {block_reason_name}")

        generated_text = ""
        if hasattr(response, 'text') and response.text is not None:
            generated_text = response.text
        elif response.candidates[0].content and response.candidates[0].content.parts:
            log.debug("Using response.candidates[0].content.parts to assemble text.")
            generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
            log.warning(f"Gemini API response for model '{model_name}' has no '.text' or usable parts. Model might have generated empty content.")

        if not generated_text.strip() and response.candidates:
            candidate = response.candidates[0]
            finish_reason_value = getattr(candidate, 'finish_reason', None)
            finish_reason_name = finish_reason_value.name if finish_reason_value else "UNKNOWN"
            log.warning(
                f"Gemini API for model '{model_name}' generated an empty or whitespace-only response. "
                f"Finish Reason: {finish_reason_name}."
            )
            if finish_reason_name not in ['STOP', 'FINISH_REASON_UNSPECIFIED']: # FINISH_REASON_UNSPECIFIED can be normal
                for rating in getattr(candidate, 'safety_ratings', []):
                    log.warning(f"Candidate Safety Rating: {rating.category.name} - {rating.probability.name}")
        return generated_text

    except genai.types.generation_types.BlockedPromptException as e:
        log.error(f"Gemini API Error (BlockedPromptException) for model '{model_name}': Prompt was blocked by safety settings. Details: {e}")
        if hasattr(e, 'response') and e.response and hasattr(e.response, 'prompt_feedback'):
            for rating in getattr(e.response.prompt_feedback, 'safety_ratings', []):
                log.error(f"Prompt Safety Rating Triggered: {rating.category.name} - {rating.probability.name}")
        raise RuntimeError(f"Gemini API prompt for '{model_name}' was blocked by safety settings: {e}") from e
    except google.auth.exceptions.RefreshError as e:
        log.error(f"Google Authentication Error (RefreshError) for model '{model_name}': {e}. This often means the API key is invalid, expired, or not properly authorized.")
        raise ValueError(f"Google Authentication failed for model '{model_name}' (RefreshError): {e}") from e
    except google.auth.exceptions.GoogleAuthError as e:
        log.error(f"Google Authentication Error for model '{model_name}': {e}. Check your API key and authentication setup.")
        raise ValueError(f"Google Authentication failed for model '{model_name}': {e}") from e
    except google_api_exceptions.PermissionDenied as e:
        log.error(f"Google API Permission Denied (403) for model '{model_name}': {e}. Ensure the API key has permissions for the Gemini API and the project is enabled/billed.")
        raise RuntimeError(f"Google API Permission Denied for model '{model_name}': {e}") from e
    except google_api_exceptions.ResourceExhausted as e:
        log.error(f"Google API Resource Exhausted (Rate Limit/Quota - 429) for model '{model_name}': {e}. You've exceeded your quota. Check Google Cloud Console.")
        raise RuntimeError(f"Google API rate limit or quota exceeded for model '{model_name}': {e}") from e
    except google_api_exceptions.InvalidArgument as e:
        log.error(f"Google API Invalid Argument (400) for model '{model_name}': {e}. This could be due to an invalid model name, incorrect request format/parameters.")
        raise ValueError(f"Invalid argument sent to Google API for model '{model_name}': {e}") from e
    except google_api_exceptions.NotFound as e:
        log.error(f"Google API Not Found (404) for model '{model_name}': {e}. The model name '{model_name}' or other resource was not found.")
        raise RuntimeError(f"Google API resource (e.g., model '{model_name}') not found: {e}") from e
    except google_api_exceptions.Aborted as e:
        log.error(f"Google API Request Aborted (409) for model '{model_name}': {e}. Often due to a concurrency issue.")
        raise RuntimeError(f"Google API request for model '{model_name}' aborted: {e}") from e
    except google_api_exceptions.Unavailable as e:
        log.error(f"Google API Service Unavailable (503) for model '{model_name}': {e}. The service is temporarily unavailable. Try again later.")
        raise RuntimeError(f"Google API service for model '{model_name}' unavailable: {e}") from e
    except google_api_exceptions.InternalServerError as e:
        log.error(f"Google API Internal Server Error (500) for model '{model_name}': {e}. An unexpected error occurred on Google's servers.")
        raise RuntimeError(f"Google API internal server error for model '{model_name}': {e}") from e
    except google_api_exceptions.DeadlineExceeded as e:
        log.error(f"Google API Deadline Exceeded (504) for model '{model_name}': {e}. The request timed out.")
        raise RuntimeError(f"Google API request for model '{model_name}' timed out: {e}") from e
    except google_api_exceptions.GoogleAPIError as e:
        log.error(f"A Google API Core Error occurred for model '{model_name}': {e}. Error Type: {type(e).__name__}")
        raise RuntimeError(f"Google API Core Error for model '{model_name}': {e}") from e
    except Exception as e:
        log.exception(f"Unexpected error during Gemini API call with model '{model_name}': {e}")
        raise RuntimeError(f"Gemini API call for '{model_name}' failed unexpectedly: {e}") from e

if TYPE_CHECKING:
    from echoscribe.pipeline import PipelineConfig
else:
    @dataclass
    class PipelineConfig:
        base_path: Path = field(default_factory=Path.cwd)

def process_with_llm(
    final_script_file: Path, output_dir: Path, llm_config: LLMConfig, pipeline_config: "PipelineConfig"
) -> None:
    log.info("Starting LLM processing with cumulative summary logic...")
    session_base_path = getattr(pipeline_config, 'base_path', Path(".")) # Expanded variable name
    session_name = session_base_path.name # Expanded variable name
    log.info(f"LLM processing for session: {session_name} (Language: {llm_config.language})")

    api_key = _load_api_key(llm_config.gemini_api_key_env_var)
    if not api_key: # Should have been caught by __post_init__ if key was mandatory there
        raise ValueError(f"Missing Gemini API key (env var: {llm_config.gemini_api_key_env_var})")

    try:
        current_session_script_content = final_script_file.read_text(encoding="utf-8")
        if not current_session_script_content.strip():
            log.warning(f"Input script '{final_script_file}' is empty. LLM results may be poor.")
    except FileNotFoundError:
        log.error(f"Final script file for LLM input not found: {final_script_file}")
        raise
    except Exception as e:
        log.exception(f"Error reading final script file {final_script_file}: {e}")
        raise IOError(f"Could not read final script file: {final_script_file}") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Previous Cumulative Summary ---
    previous_cumulative_summary_text = ""
    cumulative_summary_file_path = output_dir / llm_config.output_cumulative_meta_summary_filename
    
    if cumulative_summary_file_path.is_file():
        try:
            previous_cumulative_summary_text = cumulative_summary_file_path.read_text(encoding="utf-8")
            log.info(f"Loaded previous cumulative summary from: {cumulative_summary_file_path}")
        except Exception as e:
            log.warning(f"Could not read previous cumulative summary {cumulative_summary_file_path}: {e}. Using initial context.")
            previous_cumulative_summary_text = llm_config.initial_campaign_context
    else:
        log.info(f"No previous cumulative summary at {cumulative_summary_file_path}. Using initial context for new campaign.")
        previous_cumulative_summary_text = llm_config.initial_campaign_context

    current_session_concise_summary_text = ""

    # --- 2. Generate Current Session Concise Summary (Prompt B) ---
    if llm_config.enable_summary:
        log.info(f"Generating current session concise summary (model: {llm_config.model_name_summary})...")
        prompt_template_b = _load_prompt_template(
            DEFAULT_PROMPTS_DIR, llm_config.language, 
            llm_config.summary_prompt_filename, llm_config.fallback_summary_prompt
        )
        prompt_b_params = {
            "current_session_script": current_session_script_content,
            "cumulative_previous_sessions_summary": previous_cumulative_summary_text
        }
        try:
            formatted_prompt_b = prompt_template_b.format(**prompt_b_params)
            current_session_concise_summary_text = _call_gemini_api(
                api_key, llm_config.model_name_summary, formatted_prompt_b,
                llm_config.max_output_tokens_summary, llm_config.temperature, llm_config.safety_settings
            )
            summary_output_file = output_dir / llm_config.output_summary_filename
            summary_output_file.write_text(current_session_concise_summary_text, encoding="utf-8")
            log.info(f"Current session concise summary saved to: {summary_output_file}")
        except KeyError as e: # If prompt template has unexpected placeholders
            log.error(f"Prompt B '{DEFAULT_PROMPTS_DIR / llm_config.language / llm_config.summary_prompt_filename}' "
                      f"is missing an expected key for formatting: {e}")
        except Exception as e:
            log.error(f"Failed to generate or save current session concise summary: {e}")

    # --- 3. Generate Current Session Narrative (Prompt A) ---
    if llm_config.enable_narrative:
        log.info(f"Generating current session narrative (model: {llm_config.model_name_narrative})...")
        prompt_template_a = _load_prompt_template(
            DEFAULT_PROMPTS_DIR, llm_config.language,
            llm_config.narrative_prompt_filename, llm_config.fallback_narrative_prompt
        )
        prompt_a_params = {
            "current_session_script": current_session_script_content,
            "cumulative_previous_sessions_summary": previous_cumulative_summary_text
        }
        try:
            formatted_prompt_a = prompt_template_a.format(**prompt_a_params)
            narrative_text = _call_gemini_api(
                api_key, llm_config.model_name_narrative, formatted_prompt_a,
                llm_config.max_output_tokens_narrative, llm_config.temperature, llm_config.safety_settings
            )
            narrative_output_file = output_dir / llm_config.output_narrative_filename
            narrative_output_file.write_text(narrative_text, encoding="utf-8")
            log.info(f"Current session narrative saved to: {narrative_output_file}")
        except KeyError as e:
            log.error(f"Prompt A '{DEFAULT_PROMPTS_DIR / llm_config.language / llm_config.narrative_prompt_filename}' "
                      f"is missing an expected key for formatting: {e}")
        except Exception as e:
            log.error(f"Failed to generate or save current session narrative: {e}")

    # --- 4. Generate NEW Cumulative Meta-Summary (Prompt C) ---
    if llm_config.enable_cumulative_summary:
        if not current_session_concise_summary_text.strip():
            log.warning("Skipping cumulative meta-summary generation: current session's concise summary is empty or its generation failed.")
        else:
            log.info(f"Generating new cumulative campaign summary (model: {llm_config.model_name_summary})...")
            prompt_template_c = _load_prompt_template(
                DEFAULT_PROMPTS_DIR, llm_config.language,
                llm_config.cumulative_summary_prompt_filename, llm_config.fallback_cumulative_summary_prompt
            )
            prompt_c_params = {
                "previous_cumulative_summary": previous_cumulative_summary_text,
                "current_session_concise_summary": current_session_concise_summary_text
            }
            try:
                formatted_prompt_c = prompt_template_c.format(**prompt_c_params)
                new_cumulative_meta_summary_text = _call_gemini_api(
                    api_key, llm_config.model_name_summary, formatted_prompt_c,
                    llm_config.max_output_tokens_cumulative, llm_config.temperature, llm_config.safety_settings
                )
                
                # Backup existing cumulative summary before overwriting
                backup_cumulative_summary_path = None
                if cumulative_summary_file_path.is_file():
                    backup_cumulative_summary_path = cumulative_summary_file_path.with_suffix(cumulative_summary_file_path.suffix + ".bak")
                    try:
                        shutil.copy2(cumulative_summary_file_path, backup_cumulative_summary_path) # copy2 preserves metadata
                        log.info(f"Backed up existing cumulative summary to: {backup_cumulative_summary_path}")
                    except Exception as e_backup:
                        log.warning(f"Could not back up existing cumulative summary {cumulative_summary_file_path}: {e_backup}")
                
                # Save new_cumulative_meta_summary_text
                cumulative_summary_file_path.write_text(new_cumulative_meta_summary_text, encoding="utf-8")
                log.info(f"New cumulative campaign summary saved to: {cumulative_summary_file_path}")

            except KeyError as e:
                 log.error(f"Prompt C '{DEFAULT_PROMPTS_DIR / llm_config.language / llm_config.cumulative_summary_prompt_filename}' "
                           f"is missing an expected key for formatting: {e}")
            except Exception as e:
                log.error(f"Failed to generate or save new cumulative campaign summary: {e}")
    
    log.info("LLM processing with cumulative summary logic finished.")