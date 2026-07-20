### Sequential LLM Calls

This agent may call the LLM multiple times per case (e.g., thinking → refinement → output). Each call should be traced independently so that evaluation can measure both the intermediate reasoning and final output quality.

When you run cases, inspect the spanlog to verify that multiple spans are recorded for agents expected to do multi-turn reasoning.
