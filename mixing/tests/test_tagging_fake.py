"""The default AudioSet classifier, driven by fake ``torch`` / ``transformers`` modules (no weights)."""

import contextlib
import sys
import types

import numpy as np
import pytest

from mixing.audio import tagging as tg


class _T:
    """Minimal tensor: wraps an ndarray with the calls the classifier makes."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def to(self, dev):
        return self


def _fake_torch(*, mps=False, cuda=False):
    t = types.ModuleType("torch")
    t.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    t.no_grad = contextlib.nullcontext
    t.Tensor = type("Tensor", (), {})  # scipy's array-API shim probes torch.Tensor
    t.sigmoid = lambda x: _T(1.0 / (1.0 + np.exp(-x.a)))
    return t


LABELS = {
    "Speech": 0,
    "Narration, monologue": 1,
    "Conversation": 2,
    "Music": 3,
    "Musical instrument": 4,
    "Dog": 5,
}


class _Model:
    config = types.SimpleNamespace(label2id=LABELS)

    def eval(self):
        return self

    def to(self, dev):
        return self

    def __call__(self, **inputs):
        n = inputs["n"]
        logits = np.full((n, len(LABELS)), -10.0, dtype=np.float32)
        logits[:, 0] = 5.0  # confident speech in every window
        logits[:, 4] = 0.0  # music at 0.5
        return types.SimpleNamespace(logits=_T(logits))


class _Extractor:
    def __call__(self, chunk, *, sampling_rate, return_tensors):
        assert sampling_rate == tg.AST_SAMPLE_RATE
        return _Inputs(n=len(chunk))


class _Inputs(dict):
    def to(self, dev):
        return self


@pytest.fixture
def fakes(monkeypatch):
    tf = types.ModuleType("transformers")
    tf.AutoFeatureExtractor = types.SimpleNamespace(from_pretrained=lambda name: _Extractor())
    tf.ASTForAudioClassification = types.SimpleNamespace(from_pretrained=lambda name: _Model())
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setattr(tg, "require_ml_extra", lambda what: None)
    tg._load_ast.cache_clear()
    yield
    tg._load_ast.cache_clear()


def test_classifier_reports_speech_and_music_per_window(fakes):
    sr = tg.AST_SAMPLE_RATE
    x = np.zeros(sr * 12, dtype=np.float32)
    probs = tg.ast_audioset_classifier(device="cpu", batch_size=3)(x, sr)
    assert len(probs) == 4  # windows at 0, 2, 4, 6 s of a 12 s signal, 5 s windows
    assert probs.starts.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert np.allclose(probs.ends, [5.0, 7.0, 9.0, 11.0])
    assert np.all(probs.speech > 0.99)
    assert np.allclose(probs.music, 0.5)


def test_classifier_resamples_other_rates_and_clips_last_window(fakes):
    pytest.importorskip("scipy")
    sr = 8000
    x = np.zeros(sr * 3, dtype=np.float32)  # shorter than one window
    probs = tg.ast_audioset_classifier(device="cpu")(x, sr)
    assert len(probs) == 1 and probs.ends[0] == pytest.approx(3.0)


def test_missing_class_names_are_reported(fakes):
    clf = tg.ast_audioset_classifier(device="cpu", speech_classes=("Speech", "Nope"))
    with pytest.raises(ValueError, match=r"no AudioSet classes named \['Nope'\]"):
        clf(np.zeros(tg.AST_SAMPLE_RATE, dtype=np.float32), tg.AST_SAMPLE_RATE)


@pytest.mark.parametrize(
    "mps, cuda, expected", [(True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")]
)
def test_pick_device_prefers_mps_then_cuda(monkeypatch, mps, cuda, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=mps, cuda=cuda))
    assert tg._pick_device(None) == expected
    assert tg._pick_device("cpu") == "cpu"


def test_load_ast_is_cached(fakes):
    a = tg._load_ast("m", "cpu")
    assert tg._load_ast("m", "cpu") is a
