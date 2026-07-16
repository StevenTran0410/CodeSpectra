from agent_eval_harness.datasets.metamorphic_relations import get_relation, load_relations


def test_load_relations_parses_all_three():
    relations = load_relations()
    assert set(relations.keys()) == {
        "weakest_is_argmin",
        "injected_conflict_is_flagged",
        "empty_context_is_insufficient",
    }


def test_weakest_is_argmin_has_no_transform():
    relation = get_relation("weakest_is_argmin")
    assert relation is not None
    assert relation.transform is None
    assert relation.invariant.op == "argmin_k"
    assert relation.invariant.params["k"] == 3
    assert relation.invariant.params["allow_ties"] is True


def test_empty_context_is_insufficient_shape():
    """This is the non-CodeSpectra acceptance-target relation: transform field name is
    generic ('context'), never a CodeSpectra section letter."""
    relation = get_relation("empty_context_is_insufficient")
    assert relation is not None
    assert relation.transform.op == "degrade_section"
    assert relation.transform.params["target_field"] == "context"
    assert relation.invariant.op == "field_equals"
    assert relation.invariant.params["target_field"] == "sufficient"
    assert relation.invariant.params["expect"] is False


def test_get_relation_unknown_id_returns_none():
    assert get_relation("does_not_exist") is None
