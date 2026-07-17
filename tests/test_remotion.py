from __future__ import annotations

import json
import math
import socket
import shutil
import struct
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest
from fastapi.testclient import TestClient

from video_generator.config import active_task_ids, resolve_config
from video_generator.contracts import (
    AnchoredWord,
    AssetRights,
    BriefConstraintAssessment,
    ContentMode,
    EvidenceRecord,
    FactualResearchPack,
    MediaReference,
    OutputLanguage,
    ProbeItem,
    RemotionAsset,
    RemotionAssetBundle,
    RemotionAssetKind,
    RemotionAssetPolicy,
    RemotionEditPlan,
    RemotionEditShot,
    RemotionMotionPreset,
    RemotionShotDirection,
    RemotionSfxPreset,
    RemotionTemplate,
    RemotionTransitionPreset,
    ResearchSource,
    CreativeBrief,
    Quality,
    VisualShotMode,
    VideoStyle,
    VideoOrientation,
)
from video_generator.errors import BackendError, ConfigurationError, MediaError
from video_generator.executor import _canonicalize_host_owned_fields
from video_generator.dashboard import create_dashboard_app
from video_generator.media import MediaTools
from video_generator.net import HttpClient, HttpResponse, _AllowlistedRedirect, _SecureRedirect
from video_generator.prompting import build_frozen_assets
from video_generator.profiles import PROFILES
from video_generator.provenance import _tool_version, _tree_attestation, build_runtime_snapshot
from video_generator.remotion_assets import (
    AssetCandidate,
    PexelsClient,
    _wikimedia_rights,
    build_asset_record,
    materialize_candidate,
    search_local_media,
    write_asset_credits,
)
from video_generator.remotion_renderer import (
    _remotion_subprocess_environment,
    _run_node,
    build_remotion_manifest,
    capture_source_screenshot,
    probe_remotion_runtime,
    render_remotion_video,
    setup_remotion_runtime,
)
from video_generator.util import relative_path, sha256_file
from video_generator.run_store import RunStore
from video_generator.workflow import (
    RemotionAssetReviewDecision,
    RemotionVisualReviewBundle,
    RenderBundle,
    WorkflowEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _compact_edit_plan() -> SimpleNamespace:
    return SimpleNamespace(
        title="Fixture",
        shots=[
            SimpleNamespace(
                shot_id="shot-001",
                scene_id="scene-001",
                narration_excerpt="A fox looks into the camera.",
                template=RemotionTemplate.KINETIC_HOOK,
                headline="A fox appears",
                supporting_text="",
                body_lines=[],
                asset_kind=RemotionAssetKind.GIF,
                asset_query="quiet fox reaction",
                sfx=RemotionSfxPreset.POP,
            ),
            SimpleNamespace(
                shot_id="shot-002",
                scene_id="scene-002",
                narration_excerpt="The fox leaves.",
                template=RemotionTemplate.CONCLUSION,
                headline="The fox leaves",
                supporting_text="",
                body_lines=[],
                asset_kind=RemotionAssetKind.NONE,
                asset_query="",
                sfx=RemotionSfxPreset.NONE,
            ),
        ],
    )


def test_remotion_visual_constraints_review_the_complete_plan() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)
    calls: list[dict[str, object]] = []

    def structured_item(**kwargs):
        calls.append(kwargs["input_data"])
        return BriefConstraintAssessment(satisfied=True), []

    engine._structured_item = structured_item
    usage = engine._review_remotion_visual_constraints(
        _compact_edit_plan(),
        [
            ("must-include", 1, "one reaction GIF"),
            ("avoid", 1, "avoid stock footage"),
        ],
    )

    assert usage == []
    assert [call["constraint_kind"] for call in calls] == ["must-include", "avoid"]
    assert calls[0]["edit_plan"]["shots"][0]["asset_kind"] == "gif"
    assert "narration_excerpt" not in calls[0]["edit_plan"]["shots"][0]


@pytest.mark.parametrize("kind", ["must-include", "avoid"])
def test_remotion_visual_constraint_failure_blocks_plan_promotion(kind: str) -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(output_language=OutputLanguage.ENGLISH)

    def structured_item(**_kwargs):
        return (
            BriefConstraintAssessment(
                satisfied=False,
                scene_id="scene-001",
                evidence="The requested visual rule is not represented.",
                recommendation="Edit the opening Shot.",
            ),
            [],
        )

    engine._structured_item = structured_item
    with pytest.raises(BackendError, match="visual plan does not satisfy") as exc_info:
        engine._review_remotion_visual_constraints(
            _compact_edit_plan(),
            [(kind, 1, "one reaction GIF" if kind == "must-include" else "avoid GIFs")],
        )
    assert "dashboard" not in str(exc_info.value.action).casefold()


def test_remotion_review_failure_categories_are_strict() -> None:
    assert RemotionAssetReviewDecision(
        passed=True,
        hard_failure=False,
        failures=[],
        regeneration_instruction="",
    ).passed
    assert RemotionAssetReviewDecision(
        passed=False,
        hard_failure=True,
        failures=["The headline is clipped."],
        regeneration_instruction="",
    ).hard_failure
    assert not RemotionAssetReviewDecision(
        passed=False,
        hard_failure=False,
        failures=["The generated image is blank."],
        regeneration_instruction="Preserve the composition and generate the missing fox.",
    ).hard_failure

    with pytest.raises(ValueError, match="cannot request image regeneration"):
        RemotionAssetReviewDecision(
            passed=False,
            hard_failure=True,
            failures=["The template copy is clipped."],
            regeneration_instruction="Regenerate the fox.",
        )
    with pytest.raises(ValueError, match="requires a regeneration instruction"):
        RemotionAssetReviewDecision(
            passed=False,
            hard_failure=False,
            failures=["The asset is blank."],
            regeneration_instruction="",
        )


def test_remotion_review_checks_all_hard_failures_before_asset_regeneration() -> None:
    soft = RemotionAssetReviewDecision(
        passed=False,
        hard_failure=False,
        failures=["The generated image is blank."],
        regeneration_instruction="Generate the missing fox.",
    )
    hard = RemotionAssetReviewDecision(
        passed=False,
        hard_failure=True,
        failures=["The headline is clipped."],
        regeneration_instruction="",
    )
    soft_shot, hard_shot = _compact_edit_plan().shots

    assert WorkflowEngine._first_hard_remotion_review_failure(
        [(soft_shot, soft), (hard_shot, hard)]
    ) == (hard_shot, hard)


def test_remotion_direction_discards_fields_the_fixed_template_cannot_render() -> None:
    raw_hook = {
        "template": "kinetic_hook",
        "headline": "Cold enough to kill",
        "supporting_text": "Your thermostat would not save you",
        "body_lines": ["This is never rendered"],
        "asset_kind": "none",
        "asset_query": "also never used",
        "sfx": "whoosh",
    }
    with pytest.raises(ValueError, match="does not render body_lines"):
        RemotionShotDirection.model_validate(raw_hook)
    hook = RemotionShotDirection.model_validate(
        _canonicalize_host_owned_fields(
            "remotion_direction",
            {
                "shot_position": 1,
                "shot_count": 2,
                "narration_excerpt": "Cold enough to kill. Your thermostat would not save you.",
                "content_mode": "factual",
                "source_options": [],
            },
            raw_hook,
        )
    )
    assert hook.body_lines == []
    assert hook.asset_query == ""

    code = RemotionShotDirection.model_validate(
        _canonicalize_host_owned_fields(
            "remotion_direction",
            {
                "shot_position": 2,
                "shot_count": 4,
                "narration_excerpt": "Layer the shelter. Hide frame. Add insulation.",
                "content_mode": "factual",
                "source_options": [],
            },
            {
                "template": "code_reveal",
                "headline": "Layer the shelter",
                "supporting_text": "This template has no supporting-text slot",
                "body_lines": ["hide frame", "add insulation"],
                "asset_kind": "stock_image",
                "asset_query": "unused winter shelter",
                "sfx": "click",
            },
        )
    )
    assert code.supporting_text == ""
    assert code.asset_kind is RemotionAssetKind.NONE
    assert code.asset_query == ""
    assert code.body_lines == ["hide frame", "add insulation"]


def test_remotion_direction_keeps_meaningful_template_requirements_strict() -> None:
    with pytest.raises(ValueError, match="requires at least two body lines"):
        RemotionShotDirection.model_validate(
            {
                "template": "diagram_flow",
                "headline": "One incomplete step",
                "body_lines": ["dig"],
            }
        )

    with pytest.raises(ValueError, match="meme_cutaway requires"):
        RemotionShotDirection.model_validate(
            {
                "template": "meme_cutaway",
                "headline": "Reaction",
                "asset_kind": "none",
            }
        )


@pytest.mark.parametrize(
    ("template", "body_lines", "asset_kind", "asset_query", "sources", "expected"),
    [
        ("kinetic_hook", [], "none", "", [], "headline_zoom"),
        ("headline_zoom", [], "stock_image", "snow shelter", [], "headline_zoom"),
        (
            "source_screenshot",
            [],
            "none",
            "ignored",
            [{"source_id": "source-001"}],
            "source_screenshot",
        ),
        ("code_reveal", ["dig", "cover"], "none", "", [], "code_reveal"),
        ("diagram_flow", ["dig", "cover"], "none", "", [], "diagram_flow"),
        (
            "comparison_split",
            ["before", "after", "discarded"],
            "none",
            "",
            [],
            "comparison_split",
        ),
        ("meme_cutaway", [], "gif", "cold reaction", [], "meme_cutaway"),
        ("conclusion", [], "none", "", [], "headline_zoom"),
    ],
)
def test_remotion_direction_host_normalizes_all_fixed_template_shapes(
    template: str,
    body_lines: list[str],
    asset_kind: str,
    asset_query: str,
    sources: list[dict[str, str]],
    expected: str,
) -> None:
    normalized = _canonicalize_host_owned_fields(
        "remotion_direction",
        {
            "shot_position": 2,
            "shot_count": 4,
            "narration_excerpt": "A shelter goes from before to after when you dig and cover it.",
            "content_mode": "factual" if sources else "fiction",
            "source_options": sources,
        },
        {
            "template": template,
            "headline": "A shelter",
            "supporting_text": "A quick supporting phrase",
            "body_lines": body_lines,
            "asset_kind": asset_kind,
            "asset_query": asset_query,
            "sfx": "pop",
        },
    )
    direction = RemotionShotDirection.model_validate(normalized)

    assert direction.template.value == expected
    if expected == "source_screenshot":
        assert direction.asset_kind is RemotionAssetKind.SOURCE_SCREENSHOT
        assert direction.asset_query == ""
    if expected == "comparison_split":
        assert direction.body_lines == ["before", "after"]


@pytest.mark.parametrize(
    ("template", "body_lines", "asset_kind", "asset_query"),
    [
        ("code_reveal", ["one line"], "none", ""),
        ("meme_cutaway", [], "none", ""),
        ("source_screenshot", [], "source_screenshot", ""),
    ],
)
def test_remotion_direction_downgrades_templates_with_missing_prerequisites(
    template: str,
    body_lines: list[str],
    asset_kind: str,
    asset_query: str,
) -> None:
    normalized = _canonicalize_host_owned_fields(
        "remotion_direction",
        {
            "shot_position": 2,
            "shot_count": 4,
            "narration_excerpt": "Use the narration as the safe visible headline.",
            "content_mode": "factual",
            "source_options": [],
        },
        {
            "template": template,
            "headline": "invented visible claim",
            "supporting_text": "another invented claim",
            "body_lines": body_lines,
            "asset_kind": asset_kind,
            "asset_query": asset_query,
            "sfx": "none",
        },
    )
    direction = RemotionShotDirection.model_validate(normalized)

    assert direction.template is RemotionTemplate.HEADLINE_ZOOM
    assert direction.headline == "Use the narration as the safe visible headline."
    assert direction.supporting_text == ""
    assert direction.body_lines == []
    assert direction.asset_kind is RemotionAssetKind.NONE


def _write_tone(path: Path, *, duration_seconds: float = 4.0) -> None:
    sample_rate = 48_000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        for index in range(round(sample_rate * duration_seconds)):
            value = round(900 * math.sin(2 * math.pi * 220 * index / sample_rate))
            output.writeframesraw(struct.pack("<h", value))


def _media(path: Path, root: Path, mime_type: str) -> MediaReference:
    return MediaReference(
        path=relative_path(path, root),
        sha256=sha256_file(path),
        mime_type=mime_type,
    )


def _rights() -> AssetRights:
    return AssetRights(
        license_id="test-owned",
        license_name="Owned test fixture",
        review_status="approved",
        review_reason="Generated inside the test fixture.",
    )


def test_remotion_config_forces_cadence_and_swaps_small_tasks(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    source = source.replace('video_style = "still_image"', 'video_style = "remotion_explainer"')
    source = source.replace("offline = false", "offline = true")
    (tmp_path / "config.toml").write_text(source, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8"
    )

    config = resolve_config(tmp_path / "config.toml")
    tasks = active_task_ids(config)

    assert config.video_style is VideoStyle.REMOTION_EXPLAINER
    assert config.visual_shot_mode.value == "cadenced"
    assert config.remotion_asset_policy.value == "local_only"
    assert {
        "remotion_rhythm",
        "remotion_direction",
        "remotion_asset_select",
    } <= tasks
    assert {"visual_plan", "image_prompt_compile"}.isdisjoint(tasks)


def test_remotion_llm_schemas_exclude_host_owned_operational_fields(resolved_config) -> None:
    config = resolved_config.model_copy(
        update={
            "video_style": VideoStyle.REMOTION_EXPLAINER,
            "visual_shot_mode": VisualShotMode.CADENCED,
        }
    )
    schemas = build_frozen_assets(config)["schemas"]

    assert set(schemas["remotion_rhythm"]["properties"]) == {
        "schema_version",
        "beats",
    }
    assert set(schemas["remotion_direction"]["properties"]) == {
        "template",
        "headline",
        "supporting_text",
        "body_lines",
        "asset_kind",
        "asset_query",
        "sfx",
    }
    assert set(schemas["remotion_asset_select"]["properties"]) == {"candidate_id"}
    instructions = build_frozen_assets(config)["prompts"]["visual_review"]["instructions"]
    assert "inspect all three supplied media inputs" in instructions
    assert "inspect only the first media input" not in instructions
    assert build_frozen_assets(config)["prompts"]["remotion_rhythm"]["version"].endswith(
        ":semantic-rhythm-v1"
    )
    assert build_frozen_assets(config)["prompts"]["remotion_direction"]["version"].endswith(
        ":brief-constraints-v1"
    )
    assert build_frozen_assets(config)["prompts"]["visual_review"]["version"].endswith(
        ":remotion-hard-failure-v1"
    )


def test_source_screenshot_hosts_require_explicit_dns_allowlist(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    source = source.replace(
        "remotion_source_screenshot_hosts = []",
        'remotion_source_screenshot_hosts = ["https://example.com/path"]',
    )
    (tmp_path / "config.toml").write_text(source, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="DNS hostnames"):
        resolve_config(tmp_path / "config.toml")

    offline_source = (PROJECT_ROOT / "config.example.toml").read_text(encoding="utf-8")
    offline_source = offline_source.replace("offline = false", "offline = true")
    offline_source = offline_source.replace(
        "remotion_source_screenshot_hosts = []",
        'remotion_source_screenshot_hosts = ["example.com"]',
    )
    (tmp_path / "config.toml").write_text(offline_source, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="offline mode cannot configure"):
        resolve_config(tmp_path / "config.toml")


def test_source_options_include_only_explicitly_trusted_hosts() -> None:
    engine = object.__new__(WorkflowEngine)
    engine.config = SimpleNamespace(remotion_source_screenshot_hosts=["example.com"])
    research = FactualResearchPack(
        sources=[
            ResearchSource(source_id="source-001", url="https://docs.example.com/a", title="Allowed"),
            ResearchSource(source_id="source-002", url="https://example.net/b", title="Blocked"),
        ],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="Allowed evidence.",
                source_ids=["source-001"],
                confidence="high",
            )
        ],
    )

    assert engine._source_options(research) == [
        {"source_id": "source-001", "title": "Allowed", "publisher": ""}
    ]
    assert not engine._source_screenshot_url_allowed("ftp://docs.example.com/source")

    engine.config.offline = True
    assert not engine._source_screenshot_url_allowed("https://docs.example.com/source")


def test_local_asset_search_rejects_unrelated_files(tmp_path: Path) -> None:
    library = tmp_path / "media-library"
    library.mkdir()
    (library / "unrelated-forest.png").write_bytes(b"fixture")
    request = {
        "asset_id": "asset-001",
        "shot_id": "shot-001",
        "kind": "stock_image",
        "query": "overloaded server warning lights",
        "generated_prompt": "English fallback prompt.",
    }

    from video_generator.contracts import RemotionAssetRequest

    assert search_local_media(library, RemotionAssetRequest.model_validate(request)) == []


@pytest.mark.parametrize(
    ("suffix", "kind", "expected_mime"),
    [
        (".jpg", RemotionAssetKind.STOCK_IMAGE, "image/jpeg"),
        (".webp", RemotionAssetKind.STOCK_IMAGE, "image/webp"),
        (".mov", RemotionAssetKind.STOCK_VIDEO, "video/quicktime"),
        (".webm", RemotionAssetKind.STOCK_VIDEO, "video/webm"),
    ],
)
def test_local_asset_search_records_original_mime_type(
    tmp_path: Path,
    suffix: str,
    kind: RemotionAssetKind,
    expected_mime: str,
) -> None:
    from video_generator.contracts import RemotionAssetRequest

    library = tmp_path / "media-library"
    library.mkdir()
    (library / f"fixture{suffix}").write_bytes(b"fixture")
    request = RemotionAssetRequest(
        asset_id="asset-001",
        shot_id="shot-001",
        kind=kind,
        query="fixture",
        generated_prompt="Generated fallback.",
    )

    assert search_local_media(library, request)[0].mime_type == expected_mime


def test_empty_stock_search_falls_through_to_generated_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_generator.contracts import RemotionAssetRequest

    engine = object.__new__(WorkflowEngine)
    engine.project_root = tmp_path
    engine.config = SimpleNamespace(
        remotion_asset_policy=RemotionAssetPolicy.LOCAL_ONLY,
        output_language=OutputLanguage.ENGLISH,
        orientation=VideoOrientation.LANDSCAPE,
        remotion_allow_share_alike=False,
    )
    engine.environment = {}
    generated_asset = object()
    engine._generated_remotion_asset = lambda *_args, **_kwargs: (generated_asset, [])  # type: ignore[method-assign]
    monkeypatch.setattr("video_generator.workflow.find_asset_candidates", lambda *_args, **_kwargs: [])
    request = RemotionAssetRequest(
        asset_id="asset-001",
        shot_id="shot-001",
        kind=RemotionAssetKind.STOCK_IMAGE,
        query="overloaded server warning lights",
        generated_prompt="A clear generated fallback image.",
    )

    asset, usage, warnings = engine._resolve_remotion_asset(
        request,
        edit_plan=SimpleNamespace(),  # type: ignore[arg-type]
        work_dir=tmp_path / "asset",
        source_by_id={},
    )

    assert asset is generated_asset
    assert usage == []
    assert warnings == [
        "no approved stock candidate for shot-001; used the configured Image Backend"
    ]


def test_wikimedia_rights_are_allowlisted_and_share_alike_is_opt_in() -> None:
    cc_by = {
        "LicenseShortName": {"value": "CC BY 4.0"},
        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0/"},
        "Artist": {"value": "Fixture Creator"},
        "AttributionRequired": {"value": "true"},
    }
    by_sa = {
        **cc_by,
        "LicenseShortName": {"value": "CC BY-SA 4.0"},
        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
    }

    assert _wikimedia_rights(cc_by, allow_share_alike=False) is not None
    assert _wikimedia_rights(by_sa, allow_share_alike=False) is None
    assert _wikimedia_rights(by_sa, allow_share_alike=True).share_alike is True  # type: ignore[union-attr]
    assert _wikimedia_rights(
        {"LicenseShortName": {"value": "All rights reserved"}},
        allow_share_alike=True,
    ) is None
    assert _wikimedia_rights(
        {"LicenseShortName": {"value": "CC BY-NC 4.0"}},
        allow_share_alike=True,
    ) is None
    assert _wikimedia_rights(
        {"LicenseShortName": {"value": "CC BY-ND 4.0"}},
        allow_share_alike=True,
    ) is None
    assert _wikimedia_rights(
        {
            "LicenseShortName": {"value": "CC BY 4.0"},
            "AttributionRequired": {"value": "true"},
        },
        allow_share_alike=False,
    ) is None


def test_credits_preserve_required_attribution_and_review_notices(tmp_path: Path) -> None:
    media_path = tmp_path / "fixture.png"
    media_path.write_bytes(b"fixture")
    media = _media(media_path, tmp_path, "image/png")
    asset = RemotionAsset(
        asset_id="asset-001",
        shot_id="shot-001",
        provider="local",
        media_kind="image",
        rights=AssetRights(
            license_id="CC-BY-SA-4.0",
            license_name="Creative Commons Attribution-ShareAlike 4.0",
            attribution_required=True,
            attribution_text="Exact Creator Credit",
            share_alike=True,
            review_status="editorial_context",
            review_reason="Confirm quotation context before publication.",
        ),
        original=media,
        normalized=media,
        width=1280,
        height=720,
        transform="No transform",
        retrieved_at=datetime.now(timezone.utc),
    )

    credits_json, credits_markdown = write_asset_credits(
        [asset],
        output_dir=tmp_path / "credits",
        project_root=tmp_path,
    )

    markdown = (tmp_path / credits_markdown.path).read_text(encoding="utf-8")
    payload = json.loads((tmp_path / credits_json.path).read_text(encoding="utf-8"))
    assert "Exact Creator Credit" in markdown
    assert "ShareAlike requirements apply" in markdown
    assert "Editorial-context review required" in markdown
    assert payload["assets"][0]["attribution"] == "Exact Creator Credit"
    assert payload["assets"][0]["share_alike"] is True


def test_scene_source_grounding_maps_evidence_back_to_its_own_source() -> None:
    research = FactualResearchPack(
        sources=[
            ResearchSource(source_id="source-001", url="https://example.com/a", title="A"),
            ResearchSource(source_id="source-002", url="https://example.com/b", title="B"),
        ],
        evidence=[
            EvidenceRecord(
                evidence_id="evidence-001",
                supported_statement="The supported statement.",
                source_ids=["source-002"],
                confidence="high",
            )
        ],
    )
    grounding = {
        "supported_claims": [
            {
                "scene_id": "scene-001",
                "exact_text": "The supported statement.",
                "evidence_ids": ["evidence-001"],
            }
        ],
        "evidence_records": [],
    }

    assert WorkflowEngine._scene_grounded_source_ids(
        research,
        grounding,
        "scene-001",
    ) == ["source-002"]
    assert WorkflowEngine._scene_grounded_source_ids(
        research,
        grounding,
        "scene-002",
    ) == []


def test_factual_visible_copy_must_be_a_contiguous_spoken_span() -> None:
    excerpt = "The function disappears, but the database connection remains alive."

    assert WorkflowEngine._is_contiguous_spoken_word_span(
        "database connection remains",
        excerpt,
    )
    assert not WorkflowEngine._is_contiguous_spoken_word_span(
        "database function remains",
        excerpt,
    )


class _PexelsHttp:
    def __init__(self) -> None:
        self.url = ""

    def request(self, method: str, url: str, **kwargs: object) -> HttpResponse:
        del method, kwargs
        self.url = url
        return HttpResponse(
            200,
            {"Content-Type": "application/json"},
            json.dumps(
                {
                    "photos": [
                        {
                            "id": 7,
                            "width": 1920,
                            "height": 1080,
                            "url": "https://www.pexels.com/photo/fixture-7/",
                            "photographer": "Fixture Creator",
                            "photographer_url": "https://www.pexels.com/@fixture/",
                            "alt": "Overloaded server warning lights",
                            "src": {"large2x": "https://images.pexels.com/photos/7/fixture.jpg"},
                        }
                    ]
                }
            ).encode("utf-8"),
        )


def test_pexels_uses_english_locale_for_english_asset_queries() -> None:
    from video_generator.contracts import RemotionAssetRequest

    http = _PexelsHttp()
    request = RemotionAssetRequest(
        asset_id="asset-001",
        shot_id="shot-001",
        kind="stock_image",
        query="overloaded server warning lights",
        generated_prompt="English fallback prompt.",
    )

    candidates = PexelsClient(http=http, api_key="fixture-key").search(  # type: ignore[arg-type]
        request,
        language=OutputLanguage.FINNISH,
    )

    assert parse_qs(urlparse(http.url).query)["locale"] == ["en-US"]
    assert candidates[0].creator_name == "Fixture Creator"
    assert not ({"download_url", "source_page_url"} & candidates[0].selection_payload().keys())


def test_pexels_requests_the_selected_portrait_orientation() -> None:
    from video_generator.contracts import RemotionAssetRequest

    http = _PexelsHttp()
    request = RemotionAssetRequest(
        asset_id="asset-001",
        shot_id="shot-001",
        kind="stock_image",
        query="overloaded server warning lights",
        generated_prompt="English fallback prompt.",
    )

    PexelsClient(http=http, api_key="fixture-key").search(  # type: ignore[arg-type]
        request,
        language=OutputLanguage.ENGLISH,
        orientation=VideoOrientation.PORTRAIT,
    )

    assert parse_qs(urlparse(http.url).query)["orientation"] == ["portrait"]


def test_materialized_asset_records_the_validated_response_mime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_generator.contracts import RemotionAssetRequest

    candidate = AssetCandidate(
        candidate_id="candidate-001",
        provider="wikimedia",
        provider_asset_id="fixture",
        media_kind="image",
        mime_type="image/jpeg",
        title="Fixture",
        description="Fixture asset",
        source_page_url="https://commons.wikimedia.org/wiki/File:Fixture",
        download_url="https://upload.wikimedia.org/fixture.jpg",
        creator_name="Fixture creator",
        creator_url="",
        width=1280,
        height=720,
        duration_seconds=0,
        rights=AssetRights(
            license_id="cc0",
            license_name="CC0",
            review_status="approved",
            review_reason="Fixture",
        ),
        raw_metadata={},
    )

    class _AssetHttp:
        def request(self, *_args: object, **_kwargs: object) -> HttpResponse:
            return HttpResponse(200, {"content-type": "image/webp; charset=binary"}, b"webp")

    monkeypatch.setattr(
        "video_generator.remotion_assets.validate_public_http_url",
        lambda value: urlparse(value),
    )
    original, original_mime_type = materialize_candidate(
        candidate,
        destination_stem=tmp_path / "original",
        http=_AssetHttp(),  # type: ignore[arg-type]
    )
    normalized = tmp_path / "normalized.png"
    normalized.write_bytes(b"png")
    request = RemotionAssetRequest(
        asset_id="asset-001",
        shot_id="shot-001",
        kind=RemotionAssetKind.STOCK_IMAGE,
        query="fixture",
        generated_prompt="fixture",
    )
    asset = build_asset_record(
        request,
        candidate,
        original=original,
        original_mime_type=original_mime_type,
        normalized=normalized,
        media_kind="image",
        width=1280,
        height=720,
        duration_seconds=0,
        transform="fixture",
        project_root=tmp_path,
    )

    assert original.suffix == ".webp"
    assert original_mime_type == "image/webp"
    assert asset.original.mime_type == "image/webp"


def test_remote_asset_requires_a_supported_response_content_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = AssetCandidate(
        candidate_id="candidate-001",
        provider="wikimedia",
        provider_asset_id="fixture",
        media_kind="image",
        mime_type="image/jpeg",
        title="Fixture",
        description="Fixture asset",
        source_page_url="https://commons.wikimedia.org/wiki/File:Fixture",
        download_url="https://upload.wikimedia.org/fixture.jpg",
        creator_name="Fixture creator",
        creator_url="",
        width=1280,
        height=720,
        duration_seconds=0,
        rights=AssetRights(
            license_id="cc0",
            license_name="CC0",
            review_status="approved",
            review_reason="Fixture",
        ),
        raw_metadata={},
    )

    class _AssetHttp:
        def request(self, *_args: object, **_kwargs: object) -> HttpResponse:
            return HttpResponse(200, {}, b"not-validated")

    monkeypatch.setattr(
        "video_generator.remotion_assets.validate_public_http_url",
        lambda value: urlparse(value),
    )
    with pytest.raises(BackendError, match="did not declare a Content-Type"):
        materialize_candidate(
            candidate,
            destination_stem=tmp_path / "original",
            http=_AssetHttp(),  # type: ignore[arg-type]
        )


def test_redirect_handler_rejects_targets_outside_provider_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "video_generator.net.validate_public_http_url",
        lambda value: urlparse(value),
    )
    handler = _AllowlistedRedirect(frozenset({"images.pexels.com"}))

    with pytest.raises(BackendError, match="outside the provider host allowlist"):
        handler.redirect_request(
            Request("https://images.pexels.com/source"),
            None,
            302,
            "Found",
            {},
            "https://example.com/private-target",
        )


def test_redirect_handler_rejects_https_downgrade_before_dns_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "video_generator.net.validate_public_http_url",
        lambda _value: pytest.fail("downgrade must be rejected before DNS validation"),
    )
    handler = _AllowlistedRedirect(frozenset({"images.pexels.com"}))

    with pytest.raises(BackendError, match="may not downgrade"):
        handler.redirect_request(
            Request("https://images.pexels.com/source"),
            None,
            302,
            "Found",
            {},
            "http://images.pexels.com/target",
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("https://127.0.0.1/private", "non-public address"),
        ("https://[::1]/private", "non-public address"),
        ("https://user:secret@example.com/private", "may not contain credentials"),
        ("ftp://example.com/private", "must use public HTTP"),
    ],
)
def test_secure_redirect_rejects_nonpublic_targets(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    def _addresses(host: str, port: int) -> list[tuple[object, ...]]:
        if host == "::1":
            return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (host, port, 0, 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))]

    monkeypatch.setattr("video_generator.net.socket.getaddrinfo", _addresses)

    with pytest.raises(BackendError, match=message):
        _SecureRedirect().redirect_request(
            Request("http://provider.example/source"),
            None,
            302,
            "Found",
            {},
            target,
        )


def test_secure_redirect_strips_credentials_and_keeps_safe_negotiation_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "video_generator.net.validate_public_http_url",
        lambda value: urlparse(value),
    )
    request = Request(
        "https://images.pexels.com/source",
        headers={
            "Authorization": "Bearer secret",
            "Cookie": "session=secret",
            "X-API-Key": "secret",
            "User-Agent": "video-generator-test",
            "Accept": "image/*",
        },
    )
    redirected = _SecureRedirect().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://images.pexels.com/target",
    )
    headers = {name.casefold(): value for name, value in redirected.header_items()}

    assert headers["user-agent"] == "video-generator-test"
    assert headers["accept"] == "image/*"
    assert "authorization" not in headers
    assert "cookie" not in headers
    assert "x-api-key" not in headers
    assert any(isinstance(handler, _SecureRedirect) for handler in HttpClient()._opener.handlers)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
@pytest.mark.parametrize(
    "blocked_url",
    ["http://127.0.0.1:8765/private", "http://[fec0::1]/private"],
)
def test_screenshot_script_rejects_nonpublic_addresses_before_launch(
    tmp_path: Path,
    blocked_url: str,
) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "url": blocked_url,
                "output": str(tmp_path / "capture.png"),
                "width": 640,
                "height": 360,
                "allowedHosts": [urlparse(blocked_url).hostname],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["node", str(PROJECT_ROOT / "remotion" / "scripts" / "screenshot.mjs"), str(request)],
        cwd=PROJECT_ROOT / "remotion",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode != 0
    assert "Blocked non-public screenshot" in (completed.stderr + completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_screenshot_policy_rejects_redirect_and_subresource_hosts_before_dns() -> None:
    probe = """
import {createNetworkPolicy} from './scripts/network-policy.mjs';
const validate = createNetworkPolicy(['trusted.example']);
for (const url of ['https://redirect.example.net/next', 'https://cdn.example.net/style.css']) {
  let rejected = false;
  try { await validate(url); } catch (error) {
    rejected = String(error).includes('outside trust allowlist');
  }
  if (!rejected) process.exit(2);
}
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", probe],
        cwd=PROJECT_ROOT / "remotion",
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout


def test_source_capture_requires_prepared_runtime_without_starting_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "video_generator.remotion_renderer.probe_remotion_runtime",
        lambda _project_root: [
            ProbeItem(
                name="remotion_browser",
                ready=False,
                detail="Pinned Chrome Headless Shell is not installed",
                action="Run npm run ensure-browser in remotion/.",
            )
        ],
    )
    monkeypatch.setattr(
        "video_generator.remotion_renderer._run_node",
        lambda *_args, **_kwargs: pytest.fail("Node must not launch before readiness passes"),
    )

    with pytest.raises(MediaError, match="runtime is not ready"):
        capture_source_screenshot(
            tmp_path,
            url="https://example.com/source",
            output_path=tmp_path / "capture.png",
            width=640,
            height=360,
            allowed_hosts=["example.com"],
        )


def test_remotion_subprocesses_receive_only_allowlisted_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "remotion"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "fixture.mjs").write_text("", encoding="utf-8")
    captured: list[dict[str, str]] = []
    commands: list[list[str]] = []

    def completed(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        captured.append(dict(kwargs["env"]))  # type: ignore[arg-type]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "video_generator.remotion_renderer._ensure_remotion_project",
        lambda _project_root: root,
    )
    monkeypatch.setattr(
        "video_generator.remotion_renderer.remotion_root",
        lambda _project_root: root,
    )
    monkeypatch.setattr(
        "video_generator.remotion_renderer.shutil.which",
        lambda name: f"C:/tools/{name}.exe",
    )
    monkeypatch.setattr(
        "video_generator.remotion_renderer.probe_remotion_runtime",
        lambda _project_root: [],
    )
    monkeypatch.setattr("video_generator.remotion_renderer.subprocess.run", completed)
    environment = {
        "PATH": "C:/tools",
        "SYSTEMROOT": "C:/Windows",
        "TEMP": str(tmp_path),
        "HTTPS_PROXY": "http://proxy.example",
        "OPENAI_API_KEY": "must-not-leak",
        "PEXELS_API_KEY": "must-not-leak",
        "AWS_SECRET_ACCESS_KEY": "must-not-leak",
    }

    setup_remotion_runtime(tmp_path, download=True, environment=environment)
    _run_node(
        tmp_path,
        "fixture.mjs",
        [],
        timeout=5,
        environment=environment,
    )

    assert [command[1:] for command in commands[:4]] == [
        ["ci"],
        ["run", "build"],
        ["run", "ensure-browser"],
        ["test"],
    ]
    assert len(captured) == 5
    assert all(item["PATH"] == "C:/tools" for item in captured)
    assert all(item["HTTPS_PROXY"] == "http://proxy.example" for item in captured)
    assert all("OPENAI_API_KEY" not in item for item in captured)
    assert all("PEXELS_API_KEY" not in item for item in captured)
    assert all("AWS_SECRET_ACCESS_KEY" not in item for item in captured)
    assert _remotion_subprocess_environment(environment) == captured[0]


def test_remotion_runtime_tree_attestation_covers_transitive_and_browser_files(
    tmp_path: Path,
) -> None:
    node_modules = tmp_path / "node_modules"
    transitive = node_modules / "transitive" / "runtime.js"
    transitive.parent.mkdir(parents=True)
    transitive.write_text("version one", encoding="utf-8")
    browser = tmp_path / "chrome-headless-shell"
    browser.mkdir()
    resource = browser / "resources.pak"
    resource.write_bytes(b"browser-one")

    node_first = _tree_attestation(node_modules)
    browser_first = _tree_attestation(browser)
    transitive.write_text("version two", encoding="utf-8")
    resource.write_bytes(b"browser-two")

    assert _tree_attestation(node_modules)["tree_sha256"] != node_first["tree_sha256"]
    assert _tree_attestation(browser)["tree_sha256"] != browser_first["tree_sha256"]


def test_runtime_version_probe_uses_the_sanitized_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def completed(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs["env"])  # type: ignore[arg-type]
        return SimpleNamespace(returncode=0, stdout="v1\n", stderr="")

    monkeypatch.setattr("video_generator.provenance.subprocess.run", completed)
    environment = _remotion_subprocess_environment(
        {
            "PATH": "C:/tools",
            "SYSTEMROOT": "C:/Windows",
            "OPENAI_API_KEY": "must-not-leak",
            "PEXELS_API_KEY": "must-not-leak",
        }
    )

    assert _tool_version("node", "--version", environment=environment) == "v1"
    assert captured["PATH"] == "C:/tools"
    assert "OPENAI_API_KEY" not in captured
    assert "PEXELS_API_KEY" not in captured


def test_asset_resolver_rejects_untrusted_source_before_browser_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_generator.contracts import RemotionAssetRequest

    engine = object.__new__(WorkflowEngine)
    engine.project_root = tmp_path
    engine.config = SimpleNamespace(remotion_source_screenshot_hosts=[])
    request = RemotionAssetRequest(
        asset_id="asset-001",
        shot_id="shot-001",
        kind=RemotionAssetKind.SOURCE_SCREENSHOT,
        source_id="source-001",
    )
    source = ResearchSource(
        source_id="source-001",
        url="https://untrusted.example/source",
        title="Untrusted",
    )
    monkeypatch.setattr(
        "video_generator.workflow.capture_source_screenshot",
        lambda *_args, **_kwargs: pytest.fail("browser must not launch for an untrusted host"),
    )

    with pytest.raises(BackendError, match="not explicitly approved"):
        engine._resolve_remotion_asset(
            request,
            edit_plan=SimpleNamespace(),  # type: ignore[arg-type]
            work_dir=tmp_path / "asset",
            source_by_id={source.source_id: source},
        )


def test_asset_resolver_never_launches_browser_for_offline_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from video_generator.contracts import RemotionAssetRequest

    engine = object.__new__(WorkflowEngine)
    engine.project_root = tmp_path
    engine.config = SimpleNamespace(
        offline=True,
        remotion_source_screenshot_hosts=["example.com"],
    )
    request = RemotionAssetRequest(
        asset_id="asset-001",
        shot_id="shot-001",
        kind=RemotionAssetKind.SOURCE_SCREENSHOT,
        source_id="source-001",
    )
    source = ResearchSource(
        source_id="source-001",
        url="https://example.com/source",
        title="Trusted only when online",
    )
    monkeypatch.setattr(
        "video_generator.workflow.capture_source_screenshot",
        lambda *_args, **_kwargs: pytest.fail("offline Runs must never launch Chromium"),
    )

    with pytest.raises(BackendError, match="disabled for offline Runs"):
        engine._resolve_remotion_asset(
            request,
            edit_plan=SimpleNamespace(),  # type: ignore[arg-type]
            work_dir=tmp_path / "asset",
            source_by_id={source.source_id: source},
        )


@pytest.mark.skipif(
    shutil.which("node") is None
    or shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or not all(item.ready for item in probe_remotion_runtime(PROJECT_ROOT)),
    reason="the eight-template render requires Node, FFmpeg, and the pinned Remotion browser",
)
@pytest.mark.parametrize(("width", "height"), [(1280, 720), (720, 1280)])
def test_all_eight_templates_render_to_local_h264(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
) -> None:
    from video_generator import remotion_renderer

    monkeypatch.setattr(remotion_renderer, "remotion_root", lambda project_root: PROJECT_ROOT / "remotion")
    tools = MediaTools.discover()
    image = tmp_path / "fixture.png"
    tools.run(
        [
            tools.ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x2878a8:s={width}x{height}",
            "-frames:v",
            "1",
            str(image),
        ],
        timeout=60,
    )
    narration = tmp_path / "narration.wav"
    _write_tone(narration)
    shot_specs = [
        (RemotionTemplate.KINETIC_HOOK, RemotionAssetKind.NONE, [], RemotionMotionPreset.PUNCH_IN),
        (RemotionTemplate.HEADLINE_ZOOM, RemotionAssetKind.NONE, [], RemotionMotionPreset.SLIDE_UP),
        (RemotionTemplate.SOURCE_SCREENSHOT, RemotionAssetKind.SOURCE_SCREENSHOT, [], RemotionMotionPreset.PAN),
        (RemotionTemplate.CODE_REVEAL, RemotionAssetKind.NONE, ["const answer = 42;", "return answer;"], RemotionMotionPreset.TYPE_ON),
        (RemotionTemplate.DIAGRAM_FLOW, RemotionAssetKind.NONE, ["Input", "Change", "Result"], RemotionMotionPreset.BUILD),
        (RemotionTemplate.COMPARISON_SPLIT, RemotionAssetKind.NONE, ["Before", "After"], RemotionMotionPreset.SLIDE_UP),
        (RemotionTemplate.MEME_CUTAWAY, RemotionAssetKind.MEME, [], RemotionMotionPreset.PUNCH_IN),
        (RemotionTemplate.CONCLUSION, RemotionAssetKind.NONE, [], RemotionMotionPreset.HOLD),
    ]
    words = []
    shots = []
    assets = []
    for index, (template, kind, body_lines, motion) in enumerate(shot_specs, start=1):
        start_frame = (index - 1) * 15
        end_frame = index * 15
        word_id = f"word-{index:06d}"
        scene_id = f"scene-{index:03d}"
        shot_id = f"shot-{index:03d}"
        words.append(
            AnchoredWord(
                word_id=word_id,
                scene_id=scene_id,
                text="W" * 200 if index == 3 else f"word{index}",
                start_seconds=start_frame / 30,
                end_seconds=end_frame / 30,
            )
        )
        shots.append(
            RemotionEditShot(
                shot_id=shot_id,
                scene_id=scene_id,
                narration_excerpt=f"Narration {index}",
                start_word_id=word_id,
                end_word_id=word_id,
                start_seconds=start_frame / 30,
                end_seconds=end_frame / 30,
                start_frame=start_frame,
                end_frame=end_frame,
                template=template,
                purpose="Exercise one fixed template.",
                headline=f"Template {index}",
                supporting_text="Local deterministic render",
                body_lines=body_lines,
                asset_kind=kind,
                asset_query="reaction image" if kind is RemotionAssetKind.MEME else "",
                motion=motion,
                transition_in=(
                    RemotionTransitionPreset.SECTION_WIPE
                    if index == len(shot_specs)
                    else RemotionTransitionPreset.HARD_CUT
                ),
                sfx=RemotionSfxPreset.NONE,
            )
        )
        if kind is not RemotionAssetKind.NONE:
            media = _media(image, tmp_path, "image/png")
            assets.append(
                RemotionAsset(
                    asset_id=f"asset-{len(assets) + 1:03d}",
                    shot_id=shot_id,
                    provider="local" if kind is RemotionAssetKind.MEME else "source_screenshot",
                    media_kind="image",
                    search_query="reaction image" if kind is RemotionAssetKind.MEME else "",
                    rights=_rights(),
                    original=media,
                    normalized=media,
                    width=width,
                    height=height,
                    transform="Generated test fixture without modification",
                    retrieved_at=datetime.now(timezone.utc),
                )
            )
    edit_plan = RemotionEditPlan(
        title="Eight local templates",
        width=width,
        height=height,
        fps=30,
        duration_seconds=4,
        duration_frames=120,
        words=words,
        shots=shots,
    )
    credits_json, credits_markdown = write_asset_credits(
        assets,
        output_dir=tmp_path / "credits",
        project_root=tmp_path,
    )
    bundle = RemotionAssetBundle(
        assets=assets,
        credits_json=credits_json,
        credits_markdown=credits_markdown,
    )
    manifest = build_remotion_manifest(
        project_root=tmp_path,
        work_dir=tmp_path / "render-input",
        edit_plan=edit_plan,
        assets=bundle,
        narration_path=narration,
        output_language=OutputLanguage.FINNISH,
        music_path=None,
        captions_enabled=True,
    )
    output = tmp_path / "templates.mp4"

    render_remotion_video(
        tmp_path,
        manifest_path=manifest,
        output_path=output,
        bundle_runtime_hash="0" * 64,
    )

    probe = tools.probe_json(output)
    video = next(stream for stream in probe["streams"] if stream.get("codec_type") == "video")
    assert video["codec_name"] == "h264"
    assert int(video["width"]) == width
    assert int(video["height"]) == height
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["labels"]["takeaway"] == "TÄRKEIN AJATUS"
    assert manifest_payload["captionsEnabled"] is True
    assert len(manifest_payload["captionWords"]) == len(words)
    assert manifest_payload["shots"][-1]["transitionIn"] == "section_wipe"


@pytest.mark.skipif(
    shutil.which("node") is None
    or shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or not all(item.ready for item in probe_remotion_runtime(PROJECT_ROOT)),
    reason="the deterministic Remotion workflow requires the local renderer runtime",
)
@pytest.mark.parametrize(
    ("language", "quality", "content_mode"),
    [
        (OutputLanguage.ENGLISH, Quality.DRAFT, ContentMode.FICTION),
        (OutputLanguage.FINNISH, Quality.DRAFT, ContentMode.FICTION),
        (OutputLanguage.ENGLISH, Quality.DRAFT, ContentMode.FACTUAL),
        (OutputLanguage.ENGLISH, Quality.FINAL, ContentMode.FICTION),
    ],
)
def test_deterministic_remotion_workflow_delivers_local_english_and_finnish(
    tmp_path: Path,
    resolved_config,
    monkeypatch: pytest.MonkeyPatch,
    language: OutputLanguage,
    quality: Quality,
    content_mode: ContentMode,
) -> None:
    from video_generator import remotion_renderer

    monkeypatch.setattr(
        remotion_renderer,
        "remotion_root",
        lambda project_root: PROJECT_ROOT / "remotion",
    )
    config = resolved_config.model_copy(
        update={
            "profile": "deterministic-test",
            "project_root": str(tmp_path),
            "task_bindings": dict(PROFILES["deterministic-test"]),
            "output_language": language,
            "duration_seconds": 10,
            "quality": quality,
            "content_mode": content_mode,
            "video_style": VideoStyle.REMOTION_EXPLAINER,
            "remotion_asset_policy": RemotionAssetPolicy.LOCAL_ONLY,
            "visual_shot_mode": VisualShotMode.CADENCED,
            "visual_target_seconds": 5,
            "visual_min_seconds": 4,
            "visual_max_seconds": 8,
            "shot_target_seconds": 3,
            "shot_min_seconds": 2,
            "shot_max_seconds": 5,
            "idea_candidates": 2,
            "research_query_limit": 2 if content_mode is ContentMode.FACTUAL else 0,
            "research_source_limit": 4 if content_mode is ContentMode.FACTUAL else 0,
            "music_enabled": False,
            "captions_enabled": True,
            "animated_captions": False,
            "offline": content_mode is ContentMode.FICTION,
        }
    )
    frozen_assets = build_frozen_assets(config)
    frozen_assets["runtime_snapshot"] = build_runtime_snapshot(config)
    store = RunStore.create(
        project_root=tmp_path,
        config=config,
        brief=CreativeBrief(
            idea_direction="A tiny mystery on a snowy path",
            must_include=["one visual joke"],
        ),
        frozen_assets=frozen_assets,
    )

    with WorkflowEngine(store=store, environment={}) as workflow:
        delivery = workflow.run()

    assert delivery is not None
    assert {output.role for output in delivery.outputs} >= {
        "primary_video",
        "caption_sidecar",
        "media_credits_json",
        "media_credits_markdown",
    }
    assert all((tmp_path / output.media.path).is_file() for output in delivery.outputs)
    render_record = store.stage_record("render")
    assert render_record is not None
    rendered = store.load_artifact(render_record, RenderBundle)
    assert rendered.plan.renderer == "remotion"  # type: ignore[union-attr]
    if language is OutputLanguage.FINNISH:
        purposes = [shot.purpose for shot in rendered.plan.edit_plan.shots]  # type: ignore[union-attr]
        assert all("Make the current narration" not in purpose for purpose in purposes)
        manifest = json.loads(
            (tmp_path / rendered.plan.render_manifest.path).read_text(encoding="utf-8")  # type: ignore[union-attr]
        )
        assert manifest["labels"]["source"] == "LÄHDE"
    if quality is Quality.FINAL:
        review_record = store.stage_record("visual-review")
        assert review_record is not None
        review = store.load_artifact(review_record, RemotionVisualReviewBundle)
        assert review.reviewed is True
        report = SimpleNamespace(
            ready=True,
            model_dump=lambda *, mode: {
                "schema_version": 1,
                "ready": True,
                "profile": "deterministic-test",
                "checks": [],
                "backend_reports": [],
            },
        )
        monkeypatch.setattr(
            "video_generator.dashboard.app.verify_runtime_snapshot",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            "video_generator.dashboard.app.run_preflight",
            lambda **_kwargs: report,
        )
        monkeypatch.setattr(
            "video_generator.dashboard.app.build_runtime_snapshot",
            lambda _config: {"schema_version": 1, "snapshot_hash": "test"},
        )

        class _Supervisor:
            def __init__(self, project_root: Path) -> None:
                self.project_root = project_root
                self.jobs: dict[str, dict[str, object]] = {}

            def enqueue(self, run_id: str) -> dict[str, object]:
                job: dict[str, object] = {"run_id": run_id, "status": "queued", "pid": None}
                self.jobs[run_id] = job
                return job

            def snapshot(self, run_id: str) -> dict[str, object] | None:
                return self.jobs.get(run_id)

            def stop(self, run_id: str) -> dict[str, object] | None:
                return self.jobs.get(run_id)

            def close(self) -> None:
                pass

        original_headline = rendered.plan.edit_plan.shots[0].headline  # type: ignore[union-attr]
        approval_config = config.model_copy(
            update={"remotion_require_asset_approval": True}
        )
        approval_assets = build_frozen_assets(approval_config)
        approval_assets["runtime_snapshot"] = {
            "schema_version": 1,
            "snapshot_hash": "test",
        }
        approval_override_payload = rendered.plan.edit_plan.model_dump(mode="python")  # type: ignore[union-attr]
        approval_override_payload["shots"][0]["headline"] = "An approved edited opening"
        approval_override = RemotionEditPlan.model_validate(approval_override_payload)
        approval_assets["remotion_edit_plan_override"] = approval_override.model_dump(mode="json")
        approval_parent = RunStore.fork(
            parent=store,
            config=approval_config,
            brief=store.brief,
            frozen_assets=approval_assets,
            fork_stage="visual-review",
        )
        app = create_dashboard_app(tmp_path, supervisor_factory=_Supervisor)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            endpoint_template_response = client.post(
                f"/api/runs/{store.manifest.run_id}/shots/shot-001/fork",
                headers={"X-Dashboard-Token": app.state.dashboard_token},
                json={"template": "headline_zoom"},
            )
            source_screenshot_response = client.post(
                f"/api/runs/{store.manifest.run_id}/shots/shot-002/fork",
                headers={"X-Dashboard-Token": app.state.dashboard_token},
                json={
                    "template": "source_screenshot",
                    "asset_kind": "source_screenshot",
                    "asset_query": "",
                },
            )
            response = client.post(
                f"/api/runs/{store.manifest.run_id}/shots/shot-001/fork",
                headers={"X-Dashboard-Token": app.state.dashboard_token},
                json={"headline": "A deliberately edited opening"},
            )
            approval_response = client.post(
                f"/api/runs/{approval_parent.manifest.run_id}/approve-assets",
                headers={"X-Dashboard-Token": app.state.dashboard_token},
                json={},
            )
        assert endpoint_template_response.status_code == 422
        assert "opening Shot must keep" in endpoint_template_response.json()["detail"]
        assert source_screenshot_response.status_code == 422
        assert "no approved, scene-grounded source" in source_screenshot_response.json()[
            "detail"
        ]
        assert response.status_code == 200
        child = RunStore.open(tmp_path / "runs" / response.json()["run_id"])
        override = RemotionEditPlan.model_validate(
            child.frozen_assets["remotion_edit_plan_override"]
        )
        assert override.shots[0].headline == "A deliberately edited opening"
        assert rendered.plan.edit_plan.shots[0].headline == original_headline  # type: ignore[union-attr]
        assert approval_response.status_code == 200
        approved_child = RunStore.open(
            tmp_path / "runs" / approval_response.json()["run_id"]
        )
        assert approved_child.frozen_assets["remotion_asset_approvals"]
        preserved_override = RemotionEditPlan.model_validate(
            approved_child.frozen_assets["remotion_edit_plan_override"]
        )
        assert preserved_override.shots[0].headline == "An approved edited opening"
    store.validate_completed_outputs()
