import os
from typing import List, Dict, Any
from huggingface_hub import HfApi
from gradio_client import Client

class SpaceCandidate:
    def __init__(self, space_id: str, stage: str, zerogpu: bool, sdk: str):
        self.space_id = space_id
        self.stage = stage
        self.zerogpu = zerogpu
        self.sdk = sdk
        self.gradio_available = False
        self.usable_endpoint = None
        self.endpoint_params = []
        self.rejection_reason = None

class SpaceDiscoverer:
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.hf_token)

    def discover_spaces(self, max_inspect: int = 15) -> List[SpaceCandidate]:
        print("[DISCOVERY] Querying Hugging Face Hub for public video generation Spaces...")

        search_terms = ["ltx-video", "wan", "text-to-video", "t2v"]
        found_spaces_dict = {}

        for term in search_terms:
            try:
                results = list(self.api.list_spaces(search=term, filter="gradio", full=True, limit=15))
                for s in results:
                    if not getattr(s, "private", False):
                        found_spaces_dict[s.id] = s
            except Exception as e:
                print(f"[DISCOVERY] Error searching term '{term}': {e}")

        print(f"[DISCOVERY] Found {len(found_spaces_dict)} public Gradio Spaces.")

        candidates = []
        inspected_count = 0

        for space_id, space in found_spaces_dict.items():
            if inspected_count >= max_inspect:
                break

            runtime = getattr(space, "runtime", None)
            stage = getattr(runtime, "stage", "UNKNOWN") if runtime else "UNKNOWN"
            tags = getattr(space, "tags", []) or []
            card_data = getattr(space, "cardData", {}) or {}

            is_zerogpu = False
            if "zerogpu" in tags or card_data.get("zerogpu", False) or "zero-gpu" in tags:
                is_zerogpu = True

            candidate = SpaceCandidate(
                space_id=space_id,
                stage=stage,
                zerogpu=is_zerogpu,
                sdk=getattr(space, "sdk", "gradio")
            )

            if stage in ["STOPPED", "PAUSED", "BUILD_ERROR", "RUNTIME_ERROR"]:
                candidate.rejection_reason = f"Space runtime stage is {stage}"
                candidates.append(candidate)
                continue

            inspected_count += 1
            print(f"[DISCOVERY] Inspecting [{inspected_count}/{max_inspect}]: {space_id}...")

            try:
                client = Client(space_id, token=self.hf_token) if self.hf_token else Client(space_id)
                candidate.gradio_available = True
                api_info = client.view_api(return_format="dict")

                named_endpoints = api_info.get("named_endpoints", {})
                unnamed_endpoints = api_info.get("unnamed_endpoints", {})
                all_endpoints = {**named_endpoints, **unnamed_endpoints}

                best_ep = None
                best_params = []

                for ep_name, ep_details in all_endpoints.items():
                    parameters = ep_details.get("parameters", [])

                    has_required_file = False
                    for p in parameters:
                        p_type = str(p.get("type", "")).lower()
                        p_label = str(p.get("label", "")).lower()
                        p_name = str(p.get("parameter_name", "")).lower()

                        if p.get("required", False) and ("file" in p_type or "image" in p_label or "video" in p_label or "audio" in p_label or "image" in p_name or "video" in p_name):
                            has_required_file = True
                            break

                    if has_required_file:
                        continue

                    param_names = [str(p.get("label", "") or p.get("parameter_name", "")).lower() for p in parameters]

                    if ep_name in ["/text_to_video", "/generate_t2v", "/t2v"] or ("text" in ep_name and "video" in ep_name):
                        best_ep = ep_name
                        best_params = param_names
                        break
                    elif any("prompt" in p for p in param_names) and not best_ep:
                        best_ep = ep_name
                        best_params = param_names

                if best_ep:
                    candidate.usable_endpoint = best_ep
                    candidate.endpoint_params = best_params
                else:
                    candidate.rejection_reason = "No valid Text-to-Video endpoint found without required file inputs."

            except Exception as e:
                candidate.rejection_reason = f"Gradio client connection failed: {e}"

            candidates.append(candidate)

        return candidates
