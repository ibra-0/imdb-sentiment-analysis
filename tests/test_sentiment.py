import pytest
import torch

from sentiment import Config, TextCNN, build_vocab, clean_text, encode, predict


def test_preprocessing_matches_inference():
    assert clean_text("<BR />Great movie!!!  Isn't it?") == "great movie isn't it"


def test_unseen_validation_words_do_not_enter_training_vocabulary():
    vocab = build_vocab(["great film", "bad film"], 10)
    assert "unseen" not in vocab
    assert encode(["unseen great film extra"], vocab, 3).tolist() == [[1, vocab["great"], vocab["film"]]]
    assert encode(["great"], vocab, 3).tolist() == [[vocab["great"], 0, 0]]


def test_training_and_checkpoint_round_trip(tmp_path):
    config = Config(embedding_dim=8, num_filters=4, max_length=8, dropout=0)
    vocab = build_vocab(["great movie", "bad film"], 20)
    model = TextCNN(len(vocab), config)
    batch = encode(["great movie", "bad film"], vocab, config.max_length)
    labels = torch.tensor([1, 0])
    torch.nn.functional.cross_entropy(model(batch), labels).backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    expected = predict("great movie", model, vocab, config, torch.device("cpu"))
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = TextCNN(len(vocab), config)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert predict("great movie", restored, vocab, config, torch.device("cpu")) == expected
    with pytest.raises(ValueError):
        predict("<br> !!!", restored, vocab, config, torch.device("cpu"))
