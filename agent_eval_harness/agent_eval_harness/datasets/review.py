import json

from agent_eval_harness.store import repository


async def run_review_loop(dataset_id: str) -> None:
    print(f"=== Starting Human Review for Dataset: {dataset_id} ===")
    
    # 1. Fetch cases
    db_cases = await repository.get_dataset_cases(dataset_id)
    # Filter for those with 'synthetic' provenance
    synthetic_cases = [c for c in db_cases if c["provenance"] == "synthetic"]
    
    if not synthetic_cases:
        print("No synthetic cases left to review in this dataset.")
        return
        
    total = len(synthetic_cases)
    print(f"Found {total} cases requiring review.")
    
    for idx, db_case in enumerate(synthetic_cases, start=1):
        case_id = db_case["id"]
        
        # Load JSON fields for print
        try:
            inp = json.loads(db_case["input_json"])
        except Exception:
            inp = db_case["input_json"]
            
        try:
            exp = json.loads(db_case["expected_json"]) if db_case["expected_json"] else None
        except Exception:
            exp = db_case["expected_json"]
            
        try:
            lbl = json.loads(db_case["labels_json"]) if db_case["labels_json"] else None
        except Exception:
            lbl = db_case["labels_json"]
            
        print(f"\n--- Case {idx}/{total} (ID: {case_id}) ---")
        print(f"Input:    {json.dumps(inp, indent=2)}")
        print(f"Expected: {json.dumps(exp, indent=2)}")
        print(f"Labels:   {json.dumps(lbl, indent=2)}")
        
        while True:
            choice = input("Choice: [a]ccept / [e]dit / [r]eject / [q]uit: ").strip().lower()
            if choice in ("a", "accept"):
                await repository.update_case_provenance(case_id, "generated+reviewed")
                print("Accepted.")
                break
            elif choice in ("r", "reject"):
                await repository.delete_dataset_case(case_id)
                print("Rejected (deleted).")
                break
            elif choice in ("e", "edit"):
                print("Enter new expected output as a valid JSON string:")
                new_val = input("> ").strip()
                try:
                    # Validate JSON
                    parsed = json.loads(new_val)
                    await repository.update_case_provenance(
                        case_id, "generated+reviewed", expected_json=json.dumps(parsed)
                    )
                    print("Updated & Accepted.")
                    break
                except json.JSONDecodeError as err:
                    print(f"Invalid JSON: {err}. Please try again.")
            elif choice in ("q", "quit"):
                print("Quitting review loop.")
                return
            else:
                print("Invalid option. Please choose one of: a, e, r, q")
                
    print("\nReview session finished.")
