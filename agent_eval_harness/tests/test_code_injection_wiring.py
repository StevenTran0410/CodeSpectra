from agent_eval_harness.code_injection.wiring import build_wiring_for_codespectra
from agent_eval_harness.mapping.system_map import Component, SystemMap


def _system_map() -> SystemMap:
    return SystemMap(
        target_system_id="codespectra",
        components=[
            Component(id="project_identity", role="writer", entry_point="a:b"),
            Component(id="auditor", role="judge", entry_point="c:d"),
        ],
    )


def test_wiring_component_ids_come_from_system_map_not_guessed() -> None:
    wiring = build_wiring_for_codespectra(_system_map(), plan_id="plan-abc")

    assert wiring["component_ids"] == ["auditor", "project_identity"]
    assert wiring["plan_id"] == "plan-abc"
    assert wiring["entrypoint"]["class"] == "RunDirectorAgent"
