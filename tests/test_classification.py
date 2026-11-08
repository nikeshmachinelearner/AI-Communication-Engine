from src.classification.predict import predict_message


def test_prediction_output_structure() -> None:
    result = predict_message("Please send the report today.")

    assert "text" in result
    assert "predicted_labels" in result
    assert "scores" in result
    assert "threshold" in result

    assert isinstance(result["predicted_labels"], list)
    assert isinstance(result["scores"], dict)
    assert len(result["scores"]) == 6


def test_all_probabilities_are_valid() -> None:
    result = predict_message("Can we meet tomorrow?")

    for probability in result["scores"].values():
        assert 0.0 <= probability <= 1.0

    