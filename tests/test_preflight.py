from exotic_trainer.preflight import _encoded_length


def test_encoded_length_supports_chat_template_batch_encoding() -> None:
    assert _encoded_length({"input_ids": [[1, 2, 3, 4]]}) == 4
    assert _encoded_length([1, 2, 3]) == 3

