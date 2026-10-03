from scripts.run_bfcl_v4_benchmark import evaluate_prediction_v4


def test_irrelevance_rejects_serialized_tool_call_without_markup() -> None:
    strict, elastic = evaluate_prediction_v4(
        [["integral(x=5, y=1, function='x**3')"]],
        None,
        "BFCL_v4_irrelevance",
    )
    assert not strict
    assert not elastic


def test_irrelevance_accepts_clean_direct_answer() -> None:
    strict, elastic = evaluate_prediction_v4(
        [["The integral is 156."]],
        None,
        "BFCL_v4_irrelevance",
        attempted_tool=False,
    )
    assert strict
    assert elastic


def test_inference_error_never_passes_irrelevance() -> None:
    assert evaluate_prediction_v4(
        ["Error: request timed out"], None, "BFCL_v4_irrelevance"
    ) == (False, False)
