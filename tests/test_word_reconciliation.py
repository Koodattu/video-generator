from __future__ import annotations

import pytest

from video_generator.contracts import WordTiming
from video_generator.errors import MediaError
from video_generator.media import reconcile_word_timings


def _timings(words: list[str]) -> list[WordTiming]:
    return [
        WordTiming(
            text=word,
            start_seconds=index * 0.5,
            end_seconds=(index + 1) * 0.5,
            confidence=0.95,
        )
        for index, word in enumerate(words)
    ]


def test_reconciliation_accepts_close_finnish_words_and_split_compound() -> None:
    canonical = "Kettu astui lähemmäs narunpäähän Lumikenttä"
    recognized = _timings(["Kenitu", "astui", "lähemmässä", "narun", "päähän", "Lomikenttä"])

    words, coverage = reconcile_word_timings(
        canonical,
        recognized,
        scene_duration=3,
    )

    assert coverage == 1
    assert [word.text for word in words] == canonical.split()
    assert words[3].start_seconds == 1.5
    assert words[3].end_seconds == 2.5


def test_reconciliation_maps_recognized_compound_to_two_script_words() -> None:
    canonical = "snow field glows"
    recognized = _timings(["snowfield", "glows"])

    words, coverage = reconcile_word_timings(canonical, recognized, scene_duration=1)

    assert coverage == 1
    assert [word.text for word in words] == canonical.split()
    assert words[0].end_seconds == words[1].start_seconds


def test_reconciliation_prefers_word_match_before_compound_match() -> None:
    canonical = "Nyt portaalla näkyy nestemäistä vettä jään pinnalla."
    recognized = _timings(
        ["Nyt", "portaalla", "näkyy", "nestämäistä", "vettäjää", "pinnalla."]
    )

    words, coverage = reconcile_word_timings(canonical, recognized, scene_duration=3)

    assert coverage == 1
    assert [word.text for word in words] == canonical.split()
    assert words[4].end_seconds == words[5].start_seconds


def test_reconciliation_accepts_short_spelling_variant_with_one_real_miss() -> None:
    canonical = "As dawn breaks a single grey speck drifts into the massive approaching storm"
    recognized = _timings(
        [
            "As",
            "stone",
            "breaks",
            "a",
            "single",
            "gray",
            "speck",
            "drifts",
            "into",
            "the",
            "massive",
            "approaching",
            "storm",
        ]
    )

    words, coverage = reconcile_word_timings(canonical, recognized, scene_duration=6.5)

    assert coverage == 12 / 13
    assert [word.text for word in words] == canonical.split()


def test_reconciliation_allows_one_gap_in_a_short_scene_with_matching_token_count() -> None:
    canonical = "Keskity nyt seuraavaan konkreettiseen kohtaan."
    recognized = _timings(
        ["Esitytty", "nyt", "seuraavaan", "konkreettiseen", "kohtaan."]
    )

    words, coverage = reconcile_word_timings(canonical, recognized, scene_duration=2.5)

    assert coverage == 4 / 5
    assert [word.text for word in words] == canonical.split()
    assert words[0].confidence == 0


def test_reconciliation_recovers_split_compound_after_noisy_number_pronunciation() -> None:
    canonical = (
        "Suola liukenee veteen, jolloin ionit irtautuvat kidehilasta ja pääsevät "
        "liikkeelle. Meriveden jäätymispiste on noin −1,9 °C. Natriumkloridi "
        "liukenee veteen jopa 30-prosenttiseksi liuokseksi."
    )
    recognized = _timings(
        [
            "Suola",
            "liukenee",
            "veteen,",
            "jolloin",
            "ionit",
            "irtautuvat",
            "kidehilasta",
            "ja",
            "pääsevät",
            "liikkeelle.",
            "Meriveden",
            "jäätymispiste",
            "on",
            "noin",
            "muinaas",
            "99",
            "tigreis",
            "Celsius.",
            "Natrium",
            "-kloridi",
            "liukenee",
            "veteen",
            "jopa",
            "30",
            "-prosenttiseksi",
            "liuokseksi.",
        ]
    )

    words, coverage = reconcile_word_timings(
        canonical,
        recognized,
        scene_duration=13,
    )

    assert coverage == 20 / 22
    assert [word.text for word in words] == canonical.split()
    compound = words[16]
    assert compound.start_seconds == 9
    assert compound.end_seconds == 10


def test_reconciliation_still_rejects_unrelated_transcript() -> None:
    canonical = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
    recognized = _timings(["winter", "forest", "lantern", "quiet", "river"])

    with pytest.raises(MediaError, match="caption alignment coverage"):
        reconcile_word_timings(canonical, recognized, scene_duration=2.5)
