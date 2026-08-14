import os
from dotenv import load_dotenv
from videos.huggingface_discovery import SpaceDiscoverer

load_dotenv()

def main():
    print("==================================================")
    print("HUGGING FACE VIDEO SPACE DISCOVERY")
    print("==================================================")

    discoverer = SpaceDiscoverer()
    candidates = discoverer.discover_spaces(max_inspect=15)

    valid_candidates = []

    for c in candidates:
        print("\n--------------------------------------------------")
        print(f"Space:\n{c.space_id}")
        print(f"Status:\n{c.stage}")
        print("Public:\nYES")
        print(f"ZeroGPU:\n{'YES' if c.zerogpu else 'ZeroGPU status could not be verified'}")
        print(f"Gradio:\n{'YES' if c.gradio_available else 'NO'}")
        print(f"T2V Endpoint:\n{c.usable_endpoint or 'NONE'}")
        if c.endpoint_params:
            print(f"Parameters:\n{', '.join(c.endpoint_params)}")
        if c.rejection_reason:
            print(f"Rejection Reason:\n{c.rejection_reason}")
        else:
            valid_candidates.append(c)

    print("\n==================================================")
    print("SUMMARY")
    print("==================================================")
    print(f"Discovered inspected Spaces: {len(candidates)}")
    print(f"Valid callable T2V Candidates: {len(valid_candidates)}")

    if valid_candidates:
        print("\nTOP CANDIDATE SELECTED:")
        top = valid_candidates[0]
        print(f"  - Space: {top.space_id}")
        print(f"  - Endpoint: {top.usable_endpoint}")
        print(f"  - ZeroGPU: {'YES' if top.zerogpu else 'ZeroGPU status could not be verified'}")
    else:
        print("\n==================================================")
        print("NO USABLE FREE HUGGING FACE VIDEO SPACE FOUND")
        print("==================================================")

if __name__ == "__main__":
    main()
