"use strict";

const state = {
  bootstrap: null,
  token: "",
  runs: [],
  selectedRunId: null,
  detail: null,
  activeTab: "overview",
  eventSource: null,
  refreshTimer: null,
  preflightReady: false,
  formRevision: 0,
  preflightRevision: null,
  preflightPayload: null,
  artifactFilter: "",
};

const byId = (id) => document.getElementById(id);

function element(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined && text !== null) value.textContent = String(text);
  return value;
}

function statusClass(status) {
  return `status-${String(status || "unknown").toLowerCase().replace(/[^a-z_]/g, "")}`;
}

function statusLabel(status) {
  const labels = {
    running_external: "Running outside dashboard",
    interrupted: "Interrupted",
    stopping: "Stopping",
  };
  return labels[status] || String(status || "unknown").replaceAll("_", " ");
}

function titleCase(value) {
  return String(value || "").replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatMoney(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  return `$${numeric.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

function formatBytes(value) {
  let amount = Number(value || 0);
  const units = ["B", "KB", "MB", "GB"];
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toLocaleString(undefined, { maximumFractionDigits: unit ? 1 : 0 })} ${units[unit]}`;
}

function formatSeconds(value) {
  const seconds = Number(value || 0);
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${String(remainder).padStart(2, "0")}s`;
}

function formatDate(value, compact = false) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat(undefined, compact
    ? { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }
    : { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function listText(value) {
  if (Array.isArray(value)) return value.join(" · ");
  return value || "—";
}

async function api(path, options = {}) {
  const method = options.method || "GET";
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (method !== "GET") {
    headers.set("Content-Type", "application/json");
    headers.set("X-Dashboard-Token", state.token);
  }
  const response = await fetch(path, { ...options, method, headers });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = payload && typeof payload === "object" ? payload.detail : payload;
    const error = new Error(typeof detail === "string" ? detail : "Request failed.");
    error.detail = detail;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function toast(message, isError = false) {
  const item = element("div", `toast${isError ? " is-error" : ""}`, message);
  byId("toast-region").append(item);
  window.setTimeout(() => item.remove(), 4200);
}

function apiErrorMessage(error) {
  if (error.detail && typeof error.detail === "object") {
    if (typeof error.detail.message === "string") {
      return error.detail.action ? `${error.detail.message} ${error.detail.action}` : error.detail.message;
    }
    if (Array.isArray(error.detail.checks)) {
      const failed = error.detail.checks.filter((item) => !item.ready).map((item) => item.name);
      return failed.length ? `Preflight failed: ${failed.join(", ")}.` : "Preflight failed.";
    }
    if (Array.isArray(error.detail)) {
      return error.detail.map((item) => item.msg || "Invalid value").join(" · ");
    }
  }
  return error.message || "Something went wrong.";
}

function setLoading(message = "Loading run…") {
  const panel = element("div", "loading-panel");
  panel.append(element("span", "loading-spinner"), element("span", null, message));
  byId("workspace-content").replaceChildren(panel);
}

function setError(message) {
  const panel = element("div", "error-panel");
  panel.append(element("strong", null, "Couldn’t load this view"), element("p", null, message));
  byId("workspace-content").replaceChildren(panel);
}

function upsertSummary(summary) {
  const index = state.runs.findIndex((item) => item.run_id === summary.run_id);
  if (index >= 0) state.runs[index] = summary;
  else state.runs.unshift(summary);
  state.runs.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
}

function renderRuns() {
  const list = byId("run-list");
  const query = byId("run-filter").value.trim().toLowerCase();
  list.replaceChildren();
  const runs = state.runs.filter((run) => {
    const haystack = `${run.run_id} ${run.idea_direction} ${run.profile} ${run.status}`.toLowerCase();
    return haystack.includes(query);
  });
  byId("run-count").textContent = String(runs.length);
  for (const run of runs) {
    const button = element("button", `run-list-item${run.run_id === state.selectedRunId ? " is-active" : ""}`);
    button.type = "button";
    button.dataset.runId = run.run_id;

    const top = element("span", "run-item-top");
    top.append(element("span", `mini-status ${statusClass(run.status)}`));
    const title = element("span", "run-item-title", run.idea_direction || "Surprise me");
    top.append(title);

    const progress = element("progress", "mini-progress");
    progress.max = 1;
    progress.value = Math.max(0, Math.min(1, Number(run.progress || 0)));
    progress.setAttribute("aria-label", "Pipeline progress");

    const bottom = element("span", "run-item-bottom");
    const meta = element("span", "run-item-meta");
    meta.append(element("span", null, run.profile || "—"), element("span", null, formatDate(run.updated_at, true)));
    bottom.append(meta, element("span", "run-item-cost numeric", formatMoney(run.cost?.calculated_list_price_usd)));
    button.append(top, progress, bottom);
    button.addEventListener("click", () => selectRun(run.run_id));
    list.append(button);
  }
  if (!runs.length) {
    const empty = element("p", "run-item-meta run-list-empty", query ? "No matching runs." : "No runs yet.");
    list.append(empty);
  }
}

function renderHeader() {
  if (!state.detail) return;
  const summary = state.detail.summary;
  const status = byId("run-status");
  status.className = `status-pill ${statusClass(summary.status)}`;
  status.textContent = statusLabel(summary.status);
  byId("run-id").textContent = summary.run_id;
  byId("run-title").textContent = summary.idea_direction || "Surprise me";
  const frame = summary.delivery_width && summary.delivery_height
    ? `${titleCase(summary.orientation)} ${summary.delivery_width}×${summary.delivery_height}`
    : titleCase(summary.orientation);
  byId("run-subtitle").textContent = [summary.profile, titleCase(summary.quality), frame, summary.output_language?.toUpperCase(), `${summary.duration_seconds}s target`].filter(Boolean).join(" · ");
  byId("header-cost").textContent = formatMoney(summary.cost?.calculated_list_price_usd);
  byId("scene-count").textContent = String(state.detail.scenes?.length || 0);
  byId("artifact-count").textContent = String(state.detail.files?.length || 0);

  const resumable = ["created", "stopped", "failed", "interrupted"].includes(summary.status);
  const stoppable = ["queued", "running", "stopping"].includes(summary.status);
  byId("resume-button").classList.toggle("is-hidden", !resumable);
  byId("stop-button").classList.toggle("is-hidden", !stoppable);
}

async function refreshRuns() {
  state.runs = await api("/api/runs");
  renderRuns();
}

async function refreshDetail({ silent = false } = {}) {
  const runId = state.selectedRunId;
  if (!runId) return;
  if (!silent) setLoading();
  try {
    const detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
    if (state.selectedRunId !== runId) return;
    state.detail = detail;
    upsertSummary(detail.summary);
    renderRuns();
    renderHeader();
    renderActiveTab();
  } catch (error) {
    if (state.selectedRunId === runId) setError(apiErrorMessage(error));
  }
}

function connectRunEvents(runId) {
  if (state.eventSource) state.eventSource.close();
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  state.eventSource = source;
  source.addEventListener("run", (event) => {
    if (state.selectedRunId !== runId) return;
    let summary;
    try {
      summary = JSON.parse(event.data);
    } catch (_) {
      return;
    }
    upsertSummary(summary);
    if (state.detail) state.detail.summary = summary;
    renderRuns();
    renderHeader();
    window.clearTimeout(state.refreshTimer);
    state.refreshTimer = window.setTimeout(() => refreshDetail({ silent: true }), 180);
    if (["complete", "failed", "stopped"].includes(summary.status)) source.close();
  });
}

async function selectRun(runId) {
  state.selectedRunId = runId;
  state.detail = null;
  state.artifactFilter = "";
  byId("empty-state").classList.add("is-hidden");
  byId("run-workspace").classList.remove("is-hidden");
  renderRuns();
  setLoading();
  connectRunEvents(runId);
  await refreshDetail({ silent: true });
}

function metric(label, value, note) {
  const card = element("article", "metric-card");
  card.append(element("div", "metric-label", label), element("strong", "metric-value numeric", value));
  if (note) card.append(element("span", "metric-note", note));
  return card;
}

function panel(title, description) {
  const wrapper = element("section", "panel");
  const header = element("header", "panel-header");
  const copy = element("div");
  copy.append(element("h2", null, title));
  if (description) copy.append(element("p", null, description));
  header.append(copy);
  wrapper.append(header);
  return wrapper;
}

function emptyPanel(title, copy) {
  const value = element("div", "empty-panel");
  value.append(element("strong", null, title), element("p", null, copy));
  return value;
}

function definitionList(entries) {
  const list = element("dl", "definition-list");
  for (const [label, value] of entries) {
    const row = element("div", "definition-row");
    const description = element("dd");
    if (value instanceof Node) description.append(value);
    else description.textContent = value === undefined || value === null || value === "" ? "—" : String(value);
    row.append(element("dt", null, label), description);
    list.append(row);
  }
  return list;
}

function renderOverview() {
  const detail = state.detail;
  const summary = detail.summary;
  const root = element("div", "content-stack");
  const metrics = element("section", "metrics-grid");
  metrics.append(
    metric("Pipeline", `${summary.completed_stage_count}/${summary.total_stage_count}`, summary.current_stage ? `Now: ${titleCase(summary.current_stage)}` : "All stages accounted for"),
    metric("Calculated cost", formatMoney(detail.cost.calculated_list_price_usd), detail.cost.ledger_available ? `${detail.cost.call_count} cloud calls tracked` : "Unavailable or local-only run"),
    metric("Target duration", formatSeconds(summary.duration_seconds), `${detail.scenes.length} visual${detail.scenes.length === 1 ? "" : "s"}`),
    metric("Artifacts", String(detail.files.length), `${formatBytes(detail.files.reduce((sum, item) => sum + Number(item.size_bytes || 0), 0))} on disk`),
  );
  root.append(metrics);

  const grid = element("div", "overview-grid");
  const timelinePanel = panel("Pipeline", "The durable manifest is authoritative; fan-out item progress is folded into each stage.");
  const stages = element("div", "stage-list");
  const stageRecords = detail.manifest.stages || {};
  state.bootstrap.stages.forEach((stageName, index) => {
    const record = stageRecords[stageName] || {};
    const status = record.status || "pending";
    const row = element("div", `stage-row is-${status}`);
    row.append(
      element("span", "stage-index numeric", String(index + 1).padStart(2, "0")),
      element("span", "stage-name", titleCase(stageName)),
      element("span", "stage-backend", record.backend_id || (status === "complete" ? "local media operation" : "—")),
      element("span", `stage-status is-${status}`, status),
    );
    stages.append(row);
  });
  timelinePanel.append(stages);

  const side = element("div", "content-stack");
  const briefPanel = panel("Creative brief", "The immutable input carried into every story decision.");
  briefPanel.append(element("p", "brief-copy", detail.brief.idea_direction || "Surprise-me ideation enabled."));
  briefPanel.append(definitionList([
    ["Tone", detail.brief.tone],
    ["Themes", listText(detail.brief.themes)],
    ["Modern anchor", detail.brief.modern_anchor],
    ["Central question", detail.brief.central_question],
    ["Must include", listText(detail.brief.must_include)],
    ["Avoid", listText(detail.brief.avoid)],
  ]));
  side.append(briefPanel);

  const configPanel = panel("Run contract", "Resolved settings are frozen before execution.");
  configPanel.append(definitionList([
    ["Profile", detail.config.profile],
    ["Language", String(detail.config.output_language || "").toUpperCase()],
    ["Quality", titleCase(detail.config.quality)],
    ["Video format", `${titleCase(detail.config.orientation)} · ${detail.config.delivery_width}×${detail.config.delivery_height}`],
    ["Content", `${titleCase(detail.config.content_mode)} · ${titleCase(detail.config.content_format)}`],
    ["Narration", titleCase(detail.config.narration_pace)],
    ["Video renderer", titleCase(detail.config.video_style)],
    ["Visual cadence", titleCase(detail.config.visual_shot_mode)],
    ["Asset policy", titleCase(detail.config.remotion_asset_policy)],
    ["Cost ceiling", formatMoney(detail.config.cost_ceiling_usd, 2)],
    ["Created", formatDate(summary.created_at)],
  ]));
  side.append(configPanel);

  const warnings = summary.warnings || [];
  if (warnings.length) {
    const warningPanel = panel("Run notes", `${warnings.length} warning${warnings.length === 1 ? "" : "s"} recorded in the manifest.`);
    const list = element("ul", "warning-list");
    warnings.forEach((warning) => list.append(element("li", "warning-item", warning)));
    warningPanel.append(list);
    side.append(warningPanel);
  }
  grid.append(timelinePanel, side);
  root.append(grid);
  return root;
}

function jsonDetails(label, value) {
  const details = element("details", "detail-block");
  details.append(element("summary", null, label));
  const pre = element("pre", "code-block");
  pre.textContent = JSON.stringify(value, null, 2);
  details.append(pre);
  return details;
}

function sceneFact(label, value) {
  const fact = element("div", "scene-fact");
  fact.append(element("span", null, label), element("p", null, value || "—"));
  return fact;
}

function externalAssetLink(url) {
  if (!url) return "—";
  try {
    const parsed = new URL(url);
    if (!["http:", "https:"].includes(parsed.protocol)) return String(url);
    const link = element("a", "asset-rights-link", parsed.href);
    link.href = parsed.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  } catch (_error) {
    return String(url);
  }
}

function assetRightsPanel(asset) {
  const rights = asset?.rights || {};
  const wrapper = element("section", "asset-rights-panel");
  wrapper.append(
    element("h3", null, "Source & usage record"),
    definitionList([
      ["Source page", externalAssetLink(asset?.source_page_url)],
      ["Creator", asset?.creator_name || "—"],
      ["Creator URL", externalAssetLink(asset?.creator_url)],
      ["License", rights.license_name || rights.license_id],
      ["License URL", externalAssetLink(rights.license_url)],
      ["Provider terms", externalAssetLink(rights.terms_url)],
      ["Attribution", rights.attribution_text || (rights.attribution_required ? "Required but missing" : "Not required")],
      ["ShareAlike", rights.share_alike ? "Required" : "No"],
      ["Rights review", titleCase(rights.review_status)],
      ["Review reason", rights.review_reason],
    ]),
  );
  return wrapper;
}

function editorField(labelText, control, hint = "") {
  const label = element("label", "shot-editor-field");
  label.append(element("span", null, labelText), control);
  if (hint) label.append(element("small", null, hint));
  return label;
}

function editorSelect(value, values) {
  const select = document.createElement("select");
  for (const optionValue of values) {
    const option = document.createElement("option");
    option.value = optionValue;
    option.textContent = titleCase(optionValue);
    option.selected = optionValue === value;
    select.append(option);
  }
  return select;
}

function remotionShotEditor(scene, position, total) {
  const shot = scene.visual_brief;
  const details = element("details", "shot-editor");
  details.append(element("summary", null, "Edit this shot"));
  const intro = element("p", "shot-editor-intro", "Changes are validated and rendered in a new child Run. The parent remains untouched.");
  const grid = element("div", "shot-editor-grid");
  const sourceScreenshotEligible = state.detail.config.content_mode === "factual"
    && position > 0
    && position < total - 1
    && !state.detail.config.offline
    && (state.detail.config.remotion_source_screenshot_hosts || []).length > 0
    && (shot.source_screenshot_source_ids || []).length > 0;
  const middleTemplates = ["headline_zoom", "code_reveal", "diagram_flow", "comparison_split", "meme_cutaway"];
  if (sourceScreenshotEligible) middleTemplates.splice(1, 0, "source_screenshot");
  const templateValues = position === 0
    ? ["kinetic_hook"]
    : position === total - 1
      ? ["conclusion"]
      : middleTemplates;
  if (!templateValues.includes(shot.template)) templateValues.push(shot.template);
  const template = editorSelect(shot.template, templateValues);
  const headline = document.createElement("input");
  headline.maxLength = 80;
  headline.value = shot.headline || "";
  const supporting = document.createElement("input");
  supporting.maxLength = 160;
  supporting.value = shot.supporting_text || "";
  const bodyLines = document.createElement("textarea");
  bodyLines.rows = 4;
  bodyLines.maxLength = 700;
  bodyLines.value = (shot.body_lines || []).join("\n");
  const assetKinds = ["none", "stock_image", "stock_video", "gif", "meme", "generated_image"];
  if (sourceScreenshotEligible) assetKinds.splice(5, 0, "source_screenshot");
  if (shot.asset_kind && !assetKinds.includes(shot.asset_kind)) assetKinds.push(shot.asset_kind);
  const assetKind = editorSelect(shot.asset_kind || "none", assetKinds);
  const assetQuery = document.createElement("input");
  assetQuery.maxLength = 180;
  assetQuery.value = shot.asset_query || "";
  assetQuery.placeholder = "Concise English search phrase";
  const sfx = editorSelect(shot.sfx || "none", ["none", "click", "pop", "whoosh"]);
  grid.append(
    editorField("Template", template),
    editorField("Sound cue", sfx),
    editorField("Headline", headline, state.detail.config.content_mode === "factual" ? "Use one contiguous phrase from the spoken excerpt." : "80 characters maximum."),
    editorField("Supporting text", supporting),
    editorField("Asset type", assetKind),
    editorField("Asset query", assetQuery, "Queries are English and may be sent to enabled stock providers."),
    editorField("Template lines", bodyLines, "One line per code, diagram, or comparison item."),
  );
  const applyTemplateRules = () => {
    const selected = template.value;
    const bodyTemplate = ["code_reveal", "diagram_flow", "comparison_split"].includes(selected);
    bodyLines.disabled = !bodyTemplate;
    if (!bodyTemplate) bodyLines.value = "";
    if (selected === "source_screenshot") {
      assetKind.value = "source_screenshot";
      assetQuery.value = "";
    } else if (selected === "meme_cutaway") {
      if (!["meme", "gif", "stock_image"].includes(assetKind.value)) assetKind.value = "meme";
    } else if (!["kinetic_hook", "headline_zoom"].includes(selected)) {
      assetKind.value = "none";
      assetQuery.value = "";
    }
    const queryNeeded = !["none", "source_screenshot"].includes(assetKind.value);
    assetQuery.disabled = !queryNeeded;
    if (!queryNeeded) assetQuery.value = "";
  };
  template.addEventListener("change", applyTemplateRules);
  assetKind.addEventListener("change", applyTemplateRules);
  applyTemplateRules();
  const actions = element("div", "shot-editor-actions");
  const button = element("button", "button button-primary", "Create edited child Run");
  button.type = "button";
  button.disabled = ["queued", "running", "stopping", "running_external"].includes(state.detail.summary.status);
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Validating…";
    try {
      const payload = {
        template: template.value,
        headline: headline.value.trim(),
        supporting_text: supporting.value.trim(),
        body_lines: bodyLines.disabled ? [] : bodyLines.value.split("\n").map((line) => line.trim()).filter(Boolean),
        asset_kind: assetKind.value,
        asset_query: assetQuery.value.trim(),
        sfx: sfx.value,
      };
      const created = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/shots/${encodeURIComponent(scene.shot_id)}/fork`, {method: "POST", body: JSON.stringify(payload)});
      toast("Edited child Run created and queued.");
      await refreshRuns();
      await selectRun(created.run_id);
    } catch (error) {
      toast(apiErrorMessage(error), true);
      button.disabled = false;
    } finally {
      button.textContent = "Create edited child Run";
    }
  });
  actions.append(button);
  details.append(intro, grid, actions);
  return details;
}

async function approveSelectedAssets(button) {
  button.disabled = true;
  button.textContent = "Creating approved Run…";
  try {
    const created = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/approve-assets`, {method: "POST", body: "{}"});
    toast("Approved child Run created and queued.");
    await refreshRuns();
    await selectRun(created.run_id);
  } catch (error) {
    toast(apiErrorMessage(error), true);
    button.disabled = false;
  } finally {
    button.textContent = "Approve assets & continue";
  }
}

function renderScenes() {
  if (!state.detail.scenes.length) return emptyPanel("No visual artifacts yet", "Briefs, prompts, images, timing, and reviews will join here by stable Scene or Shot ID as the run advances.");
  const workspace = element("div", "visual-workspace");
  const remotionScenes = state.detail.scenes.filter((scene) => Boolean(scene.visual_brief?.template));
  if (remotionScenes.length) {
    const timeline = element("div", "shot-timeline");
    timeline.setAttribute("aria-label", "Word-anchored shot timeline");
    for (const scene of remotionScenes) {
      const segment = element("button", `shot-segment template-${scene.visual_brief.template}`);
      segment.type = "button";
      const duration = Math.max(0.1, Number(scene.timing?.end_seconds || 0) - Number(scene.timing?.start_seconds || 0));
      segment.classList.add(`duration-${Math.min(12, Math.max(1, Math.round(duration)))}`);
      segment.title = `${scene.shot_id}: ${scene.visual_brief.headline}`;
      segment.append(element("strong", "numeric", scene.shot_id), element("span", null, scene.visual_brief.headline));
      segment.addEventListener("click", () => document.getElementById(`visual-${scene.shot_id}`)?.scrollIntoView({behavior: "smooth", block: "center"}));
      timeline.append(segment);
    }
    workspace.append(timeline);
  }
  const approval = state.detail.asset_approval;
  if (approval?.required && approval.asset_count) {
    const bar = element("section", `asset-approval${approval.approved ? " is-approved" : ""}`);
    const copy = element("div");
    copy.append(
      element("strong", null, approval.approved ? "Current assets approved" : "Media approval required"),
      element("p", null, approval.approved ? "This child Run contains approval hashes for every current asset." : "Review every visual, source page, license, and warning below before continuing."),
    );
    bar.append(copy);
    if (!approval.approved) {
      const confirm = element("label", "approval-confirm");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      confirm.append(checkbox, element("span", null, "I reviewed all current assets and usage terms."));
      const button = element("button", "button button-primary", "Approve assets & continue");
      button.type = "button";
      button.disabled = true;
      checkbox.addEventListener("change", () => { button.disabled = !checkbox.checked; });
      button.addEventListener("click", () => approveSelectedAssets(button));
      bar.append(confirm, button);
    }
    workspace.append(bar);
  }
  const grid = element("div", "scene-grid");
  const deliveryWidth = Number(state.detail.config.delivery_width || 16);
  const deliveryHeight = Number(state.detail.config.delivery_height || 9);
  const portraitFrame = deliveryHeight > deliveryWidth;
  grid.classList.toggle("is-portrait", portraitFrame);
  for (const [position, scene] of state.detail.scenes.entries()) {
    const visualId = scene.shot_id || scene.scene_id;
    const card = element("article", "scene-card");
    card.id = `visual-${visualId}`;
    const imageWrap = element("div", "scene-image-wrap");
    imageWrap.classList.add(portraitFrame ? "is-portrait" : "is-landscape");
    if (scene.image_url) {
      if (scene.media_kind === "video") {
        const video = document.createElement("video");
        video.src = scene.image_url;
        video.controls = true;
        video.loop = true;
        video.muted = true;
        video.preload = "metadata";
        imageWrap.append(video);
      } else {
        const image = document.createElement("img");
        image.src = scene.image_url;
        image.alt = `Visual asset for ${visualId}`;
        image.loading = "lazy";
        imageWrap.append(image);
      }
    } else {
      imageWrap.append(element("div", "scene-image-placeholder", "Image not generated yet"));
    }
    imageWrap.append(element("span", "scene-badge numeric", visualId));

    const body = element("div", "scene-card-body");
    const heading = element("div", "scene-card-heading");
    heading.append(element("h2", null, titleCase(visualId)));
    if (scene.timing) {
      heading.append(element("span", "scene-time numeric", `${formatSeconds(scene.timing.start_seconds)} → ${formatSeconds(scene.timing.end_seconds)}`));
    }
    body.append(heading);
    if (scene.script?.spoken_text) body.append(element("p", "scene-script", scene.script.spoken_text));

    const facts = element("div", "scene-facts");
    const isRemotion = Boolean(scene.visual_brief?.template);
    facts.append(
      sceneFact(isRemotion ? "Template" : "Visible action", scene.visual_brief?.template || scene.visual_brief?.action),
      sceneFact(isRemotion ? "Visual purpose" : "Story moment", scene.visual_brief?.purpose || scene.visual_brief?.story_moment),
      sceneFact("Parent section", scene.shot_id ? scene.scene_id : null),
      sceneFact(isRemotion ? "Asset provider" : "Incoming continuity", scene.asset?.provider || listText(scene.visual_brief?.continuity_from_previous)),
      sceneFact(isRemotion ? "License" : "State after", scene.asset?.rights?.license_name || listText(scene.visual_brief?.state_after_scene)),
    );
    if (isRemotion) facts.append(sceneFact("Rights review", titleCase(scene.asset?.rights?.review_status)));
    body.append(facts);
    if (isRemotion && scene.asset?.warnings?.length) {
      const warnings = element("ul", "warning-list");
      for (const warning of scene.asset.warnings) warnings.append(element("li", "warning-item", warning));
      body.append(warnings);
    }
    if (isRemotion && scene.asset) body.append(assetRightsPanel(scene.asset));
    if (scene.audio_url) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = scene.audio_url;
      audio.className = "scene-audio";
      body.append(audio);
    }
    if (scene.review) {
      const review = element("div", `review-strip${scene.review.passed ? "" : " is-failed"}`);
      review.append(element("span", null, scene.review.passed ? "Visual review passed" : "Visual review failed"));
      const scores = scene.review.scores ? Object.values(scene.review.scores) : [];
      review.append(element("span", "numeric", scores.length ? `min ${Math.min(...scores)}/5` : "reviewed"));
      body.append(review);
    }
    if (scene.visual_brief) body.append(jsonDetails("Visual brief", scene.visual_brief));
    if (scene.image_request) body.append(jsonDetails("Compiled image request", scene.image_request));
    if (isRemotion) body.append(remotionShotEditor(scene, position, state.detail.scenes.length));
    card.append(imageWrap, body);
    grid.append(card);
  }
  workspace.append(grid);
  return workspace;
}

function renderOutputs() {
  const outputs = state.detail.outputs || [];
  if (!outputs.length) return emptyPanel("No delivery outputs yet", "Finished video, caption sidecars, and delivery manifests appear here after rendering.");
  const grid = element("div", "output-grid");
  for (const file of outputs) {
    const card = element("article", "output-card");
    const preview = element("div", "output-preview");
    if (file.mime_type.startsWith("video/")) {
      const media = document.createElement("video");
      media.controls = true;
      media.preload = "metadata";
      media.src = file.url;
      preview.append(media);
    } else if (file.mime_type.startsWith("audio/")) {
      const media = document.createElement("audio");
      media.controls = true;
      media.preload = "metadata";
      media.src = file.url;
      preview.append(media);
    } else if (file.mime_type.startsWith("image/")) {
      const media = document.createElement("img");
      media.src = file.url;
      media.alt = file.name;
      media.loading = "lazy";
      preview.append(media);
    } else {
      preview.append(element("span", "output-file-icon", file.name.split(".").pop()?.toUpperCase() || "FILE"));
    }
    const meta = element("div", "output-meta");
    const copy = element("div");
    copy.append(element("strong", null, file.name), element("small", "numeric", `${formatBytes(file.size_bytes)} · ${file.mime_type}`));
    const link = element("a", "file-link", "Open");
    link.href = file.url;
    link.target = "_blank";
    link.rel = "noopener";
    meta.append(copy, link);
    card.append(preview, meta);
    grid.append(card);
  }
  return grid;
}

function createTable(headers) {
  const wrap = element("div", "table-wrap");
  const table = element("table", "data-table");
  const head = element("thead");
  const row = element("tr");
  headers.forEach((label) => row.append(element("th", null, label)));
  head.append(row);
  const body = element("tbody");
  table.append(head, body);
  wrap.append(table);
  return { wrap, body };
}

function renderCosts() {
  const cost = state.detail.cost;
  const root = element("div", "content-stack");
  const metrics = element("section", "metrics-grid");
  metrics.append(
    metric("Calculated list price", formatMoney(cost.calculated_list_price_usd), "Known direct + inherited usage × frozen rates"),
    metric("Direct list price", formatMoney(cost.direct_calculated_list_price_usd), "Calls incurred by this Run only"),
    metric("Provider reported", formatMoney(cost.provider_reported_usd), "Only when the API reports a billed amount"),
    metric("Unresolved maximum", formatMoney(cost.unresolved_maximum_usd), "Current-Run calls whose billing outcome is unknown"),
    metric("Conservative reserved", formatMoney(cost.conservative_reserved_usd), "Hard ceiling consumption; not spend"),
  );
  root.append(metrics, element("p", "cost-note", cost.label));

  const breakdownPanel = panel("Cost by model and task", "Fan-out usage is counted once from the provider-call ledger.");
  if (cost.breakdown.length) {
    const table = createTable(["Backend", "Task", "Calls", "Inherited", "Calculated", "Reported", "Unresolved max"]);
    cost.breakdown.forEach((item) => {
      const row = element("tr");
      row.append(
        element("td", "path-cell", item.backend_id),
        element("td", null, titleCase(item.task_id)),
        element("td", "numeric", item.calls),
        element("td", "numeric", item.inherited_calls),
        element("td", "numeric", formatMoney(item.estimated_usd)),
        element("td", "numeric", formatMoney(item.actual_usd)),
        element("td", "numeric", formatMoney(item.unresolved_usd)),
      );
      table.body.append(row);
    });
    breakdownPanel.append(table.wrap);
  } else {
    breakdownPanel.append(emptyPanel("No priced cloud usage", "This run is local-only, legacy, or has not reached a cloud call yet."));
  }
  root.append(breakdownPanel);

  const calls = state.detail.manifest.cloud_calls || [];
  if (calls.length) {
    const callsPanel = panel("Provider-call ledger", "Written before every call and settled immediately when usage returns.");
    const table = createTable(["Status", "Task", "Backend", "Origin", "Calculated", "Reserved", "Elapsed", "Request ID"]);
    calls.forEach((call) => {
      const row = element("tr");
      row.append(
        element("td", `cost-status is-${call.status}`, call.status),
        element("td", null, titleCase(call.task_id)),
        element("td", "path-cell", call.backend_id),
        element("td", null, call.inherited ? "parent Run" : "this Run"),
        element("td", "numeric", call.estimated_usd === null || call.estimated_usd === undefined ? "unpriced" : formatMoney(call.estimated_usd)),
        element("td", "numeric", formatMoney(call.reserved_usd)),
        element("td", "numeric", formatSeconds(call.elapsed_seconds || 0)),
        element("td", "path-cell", call.provider_request_id || "—"),
      );
      table.body.append(row);
    });
    callsPanel.append(table.wrap);
    root.append(callsPanel);
  }
  return root;
}

function renderArtifacts() {
  const root = element("section", "panel");
  const header = element("header", "panel-header");
  const copy = element("div");
  copy.append(element("h2", null, "Run Bundle"), element("p", null, "Every generated, frozen, logged, and delivered file under this Run."));
  header.append(copy);
  root.append(header);

  const toolbar = element("div", "toolbar");
  const filter = document.createElement("input");
  filter.type = "search";
  filter.placeholder = "Filter paths or MIME types";
  filter.value = state.artifactFilter;
  filter.addEventListener("input", () => {
    state.artifactFilter = filter.value;
    updateArtifactRows(body, count);
  });
  const count = element("span", "metric-note numeric");
  toolbar.append(filter, count);
  root.append(toolbar);

  const table = createTable(["Path", "Type", "Size", "Provenance", "Modified"]);
  const body = table.body;
  root.append(table.wrap);
  updateArtifactRows(body, count);
  return root;
}

function updateArtifactRows(body, count) {
  const query = state.artifactFilter.trim().toLowerCase();
  const files = state.detail.files.filter((file) => `${file.path} ${file.mime_type}`.toLowerCase().includes(query));
  count.textContent = `${files.length} of ${state.detail.files.length} files`;
  body.replaceChildren();
  files.forEach((file) => {
    const row = element("tr");
    const pathCell = element("td", "path-cell");
    const link = element("a", null, file.path);
    link.href = file.url;
    link.target = "_blank";
    link.rel = "noopener";
    pathCell.append(link);
    row.append(
      pathCell,
      element("td", null, file.mime_type),
      element("td", "numeric", formatBytes(file.size_bytes)),
    );
    const hashCell = element("td");
    hashCell.append(element(
      "span",
      `hash-badge${file.hash_recorded ? "" : " is-unmanaged"}`,
      file.hash_recorded ? "hash recorded" : "internal / untracked",
    ));
    row.append(hashCell, element("td", "numeric", formatDate(Number(file.modified_at) * 1000, true)));
    body.append(row);
  });
}

function renderActiveTab() {
  if (!state.detail) return;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === state.activeTab));
  let content;
  if (state.activeTab === "scenes") content = renderScenes();
  else if (state.activeTab === "outputs") content = renderOutputs();
  else if (state.activeTab === "costs") content = renderCosts();
  else if (state.activeTab === "artifacts") content = renderArtifacts();
  else content = renderOverview();
  const workspace = byId("workspace-content");
  workspace.replaceChildren(content);
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    workspace.animate([{ opacity: 0, transform: "translateY(3px)" }, { opacity: 1, transform: "translateY(0)" }], { duration: 130, easing: "ease-out" });
  }
}

function splitList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean).filter((item, index, values) => values.indexOf(item) === index);
}

function formPayload() {
  const taskOverrides = {};
  document.querySelectorAll("[data-model-task]").forEach((select) => {
    taskOverrides[select.dataset.modelTask] = select.value;
  });
  const defaults = state.bootstrap.defaults;
  const surpriseMe = byId("surprise-me").checked;
  return {
    brief: {
      schema_version: 1,
      idea_direction: surpriseMe ? "" : byId("idea-direction").value.trim(),
      surprise_me: surpriseMe,
      tone: byId("tone").value.trim(),
      themes: splitList(byId("themes").value),
      must_include: splitList(byId("must-include").value),
      avoid: splitList(byId("avoid").value),
      research_focus: splitList(byId("research-focus").value),
      modern_anchor: byId("modern-anchor").value.trim(),
      central_question: byId("central-question").value.trim(),
      misconception: byId("misconception").value.trim(),
      desired_takeaway: byId("desired-takeaway").value.trim(),
    },
    options: {
      profile: byId("profile").value,
      output_language: byId("output-language").value,
      duration_seconds: Number(byId("duration-seconds").value),
      quality: byId("quality").value,
      orientation: byId("orientation").value,
      content_mode: byId("content-mode").value,
      content_format: byId("content-format").value,
      narration_pace: byId("narration-pace").value,
      narration_delivery: byId("narration-delivery").value.trim(),
      video_style: byId("video-style").value,
      style: byId("style-id").value.trim(),
      style_description: byId("style-description").value.trim(),
      remotion_asset_policy: byId("remotion-asset-policy").value,
      remotion_allow_share_alike: byId("remotion-share-alike").checked,
      remotion_require_asset_approval: byId("remotion-asset-approval").checked,
      remotion_source_screenshot_hosts: splitList(byId("remotion-screenshot-hosts").value),
      offline: byId("offline").checked,
      cost_ceiling_usd: Number(byId("cost-ceiling").value),
      idea_candidates: Number(byId("idea-candidates").value),
      research_query_limit: Number(byId("research-queries").value),
      research_source_limit: Number(byId("research-sources").value),
      visual_target_seconds: Number(byId("visual-target").value),
      visual_min_seconds: Number(defaults.visual_min_seconds),
      visual_max_seconds: Number(defaults.visual_max_seconds),
      visual_shot_mode: byId("visual-shot-mode").value,
      shot_target_seconds: Number(byId("shot-target").value),
      shot_min_seconds: Number(defaults.shot_min_seconds),
      shot_max_seconds: Number(defaults.shot_max_seconds),
      music_enabled: byId("music-enabled").checked,
      captions_enabled: byId("captions-enabled").checked,
      animated_captions: byId("animated-captions").checked,
      task_overrides: taskOverrides,
    },
  };
}

function invalidatePreflight() {
  state.formRevision += 1;
  state.preflightReady = false;
  state.preflightRevision = null;
  state.preflightPayload = null;
  byId("start-run-button").disabled = true;
  byId("preflight-panel").classList.add("is-hidden");
  byId("form-status").textContent = "Run preflight after changes. It is local and will not call paid cloud models.";
  syncActiveTasks();
}

function applyModelBindings(bindings) {
  for (const [taskId, backendId] of Object.entries(bindings)) {
    const select = document.querySelector(`[data-model-task="${taskId}"]`);
    if (select && [...select.options].some((option) => option.value === backendId)) select.value = backendId;
  }
}

function applyProfileModels() {
  applyModelBindings(state.bootstrap.profiles[byId("profile").value] || {});
}

function syncActiveTasks() {
  const quality = byId("quality").value;
  const music = byId("music-enabled").checked;
  const captions = byId("captions-enabled").checked;
  const remotion = byId("video-style").value === "remotion_explainer";
  if (remotion) byId("visual-shot-mode").value = "cadenced";
  byId("visual-shot-mode").disabled = remotion;
  byId("remotion-asset-policy").disabled = !remotion;
  byId("remotion-share-alike").disabled = !remotion;
  byId("remotion-asset-approval").disabled = !remotion;
  const cadencedVisuals = remotion || byId("visual-shot-mode").value === "cadenced";
  const factual = byId("content-mode").value === "factual";
  const offline = byId("offline").checked;
  const research = Number(byId("research-queries").value || 0) > 0;
  document.querySelectorAll(".model-task").forEach((row) => {
    const task = row.dataset.taskId;
    const inactive = (["claim_inventory", "factual_review"].includes(task) && !factual)
      || (task === "visual_review" && quality !== "final")
      || (["music_brief", "music_generate"].includes(task) && !music)
      || (task === "caption_alignment" && !captions && !cadencedVisuals)
      || (["remotion_direction", "remotion_asset_select"].includes(task) && !remotion)
      || (["visual_plan", "image_prompt_compile"].includes(task) && remotion)
      || (task === "search" && (offline || !research));
    row.classList.toggle("is-inactive", inactive);
  });
}

function renderModelGrid() {
  const container = byId("model-grid");
  container.replaceChildren();
  for (const [groupName, taskIds] of Object.entries(state.bootstrap.task_groups)) {
    const group = element("section", "model-group");
    const header = element("header", "model-group-header");
    header.append(element("span", null, groupName), element("span", "numeric", `${taskIds.length} tasks`));
    const list = element("div", "model-task-list");
    for (const taskId of taskIds) {
      const task = state.bootstrap.tasks.find((item) => item.task_id === taskId);
      if (!task) continue;
      const row = element("label", "model-task");
      row.dataset.taskId = taskId;
      const label = element("span", "model-task-label");
      label.append(element("strong", null, titleCase(taskId)), element("small", null, task.protocol));
      const select = document.createElement("select");
      select.dataset.modelTask = taskId;
      task.backend_options.forEach((backendId) => {
        const backend = state.bootstrap.backends[backendId];
        const option = document.createElement("option");
        const tierLabel = {
          preferred: "recommended",
          alternative: "alternative",
          experimental: "experimental",
          legacy: "legacy · lower quality",
        }[backend.selection_tier];
        option.value = backendId;
        option.textContent = `${backend.cloud ? "☁" : "◆"} ${backend.provider} · ${backend.model_id}${tierLabel ? ` · ${tierLabel}` : ""}${backend.cloud && !backend.configured ? " · key missing" : ""}`;
        option.title = backend.notes || "";
        select.append(option);
      });
      select.addEventListener("change", invalidatePreflight);
      row.append(label, select);
      list.append(row);
    }
    group.append(header, list);
    container.append(group);
  }
  applyProfileModels();
  syncActiveTasks();
}

function applyDefaults() {
  const defaults = state.bootstrap.defaults;
  byId("profile").replaceChildren();
  Object.keys(state.bootstrap.profiles).forEach((profileName) => {
    const option = element("option", null, titleCase(profileName));
    option.value = profileName;
    byId("profile").append(option);
  });
  byId("profile").value = defaults.profile;
  byId("output-language").value = defaults.output_language;
  byId("quality").value = defaults.quality;
  byId("orientation").value = defaults.orientation || "landscape";
  byId("content-mode").value = defaults.content_mode;
  byId("content-format").value = defaults.content_format;
  byId("narration-pace").value = defaults.narration_pace;
  byId("narration-delivery").value = defaults.narration_delivery || "";
  byId("video-style").value = defaults.video_style || "still_image";
  byId("remotion-asset-policy").value = defaults.remotion_asset_policy || "stock_preferred";
  byId("duration-seconds").value = defaults.duration_seconds;
  byId("cost-ceiling").value = defaults.cost_ceiling_usd;
  byId("idea-candidates").value = defaults.idea_candidates;
  byId("visual-target").value = defaults.visual_target_seconds;
  byId("visual-target").min = defaults.visual_min_seconds;
  byId("visual-target").max = defaults.visual_max_seconds;
  byId("visual-shot-mode").value = defaults.visual_shot_mode;
  byId("shot-target").value = defaults.shot_target_seconds;
  byId("shot-target").min = defaults.shot_min_seconds;
  byId("shot-target").max = defaults.shot_max_seconds;
  byId("research-queries").value = defaults.research_query_limit;
  byId("research-sources").value = defaults.research_source_limit;
  byId("style-description").value = defaults.style_description || "";
  byId("remotion-screenshot-hosts").value = (defaults.remotion_source_screenshot_hosts || []).join(", ");
  byId("style-id").value = defaults.style;
  byId("offline").checked = Boolean(defaults.offline);
  byId("captions-enabled").checked = Boolean(defaults.captions_enabled);
  byId("animated-captions").checked = Boolean(defaults.animated_captions);
  byId("music-enabled").checked = Boolean(defaults.music_enabled);
  byId("remotion-share-alike").checked = Boolean(defaults.remotion_allow_share_alike);
  byId("remotion-asset-approval").checked = Boolean(defaults.remotion_require_asset_approval);
  renderModelGrid();
  applyModelBindings(state.bootstrap.default_task_bindings || {});
}

function validateStoryDirection() {
  const direction = byId("idea-direction");
  if (!direction.value.trim() && !byId("surprise-me").checked) {
    direction.setCustomValidity("Add an idea direction or enable Surprise me.");
    direction.reportValidity();
    return false;
  }
  direction.setCustomValidity("");
  const contentMode = byId("content-mode");
  const mythbusterNeedsFacts = byId("content-format").value === "mythbuster" && contentMode.value !== "factual";
  const factualNeedsResearch = contentMode.value === "factual" && (
    byId("offline").checked
    || Number(byId("research-queries").value || 0) === 0
    || Number(byId("research-sources").value || 0) === 0
  );
  contentMode.setCustomValidity(
    mythbusterNeedsFacts
      ? "Myth-buster format requires factual content mode."
      : factualNeedsResearch
        ? "Factual content requires live research with nonzero query and source limits."
        : "",
  );
  if (mythbusterNeedsFacts || factualNeedsResearch) {
    contentMode.reportValidity();
    return false;
  }
  const shotTarget = byId("shot-target");
  const sceneCount = Math.max(1, Math.ceil(
    Number(byId("duration-seconds").value) / Number(byId("visual-target").value),
  )) + 1;
  const estimatedShots = Math.max(
    sceneCount,
    Math.ceil(Number(byId("duration-seconds").value) / Number(shotTarget.value))
      + Math.ceil(sceneCount / 2),
  );
  const oversizedCadence = byId("visual-shot-mode").value === "cadenced" && estimatedShots > 72;
  shotTarget.setCustomValidity(
    oversizedCadence
      ? `This cadence estimates ${estimatedShots} images; the current limit is 72.`
      : "",
  );
  if (oversizedCadence) {
    shotTarget.reportValidity();
    return false;
  }
  return byId("new-run-form").reportValidity();
}

function renderPreflight(report, payload, revision) {
  const panel = byId("preflight-panel");
  panel.classList.remove("is-hidden");
  panel.replaceChildren();
  const summary = element("div", "preflight-summary");
  summary.append(
    element("strong", null, report.ready ? "Ready to run" : "Action required"),
    element("span", "numeric", `${formatMoney(report.cost.projected_total_usd, 2)} projected reservation · ${report.cost.scene_count} sections · ${report.cost.visual_shot_count || report.cost.scene_count} images`),
  );
  panel.append(summary);
  const checks = element("div", "preflight-checks");
  const values = [...report.checks];
  report.backend_reports.forEach((backend) => {
    values.push({ name: backend.backend_id, ready: backend.ready, detail: backend.items.map((item) => item.detail).join(" · ") });
  });
  values.forEach((check) => {
    const item = element("div", `preflight-check${check.ready ? "" : " is-failed"}`);
    item.append(element("span", "check-mark", check.ready ? "✓" : "!"));
    const copy = element("span");
    copy.append(element("strong", null, titleCase(check.name)), element("small", null, check.ready ? check.detail : `${check.detail}${check.action ? ` · ${check.action}` : ""}`));
    item.append(copy);
    checks.append(item);
  });
  panel.append(checks);
  state.preflightReady = Boolean(report.ready);
  state.preflightRevision = report.ready ? revision : null;
  state.preflightPayload = report.ready ? payload : null;
  byId("start-run-button").disabled = !state.preflightReady;
  byId("form-status").textContent = report.ready
    ? "Preflight passed. Starting creates a durable Run Bundle and queues one isolated worker process."
    : "Resolve the failed checks or change the model selections, then run preflight again.";
}

async function runPreflight() {
  if (!validateStoryDirection()) return;
  const button = byId("preflight-button");
  const payload = formPayload();
  const revision = state.formRevision;
  button.disabled = true;
  button.textContent = "Checking…";
  try {
    const report = await api("/api/preflight", { method: "POST", body: JSON.stringify(payload) });
    if (revision !== state.formRevision) return;
    renderPreflight(report, payload, revision);
  } catch (error) {
    if (revision !== state.formRevision) return;
    state.preflightReady = false;
    state.preflightRevision = null;
    state.preflightPayload = null;
    byId("start-run-button").disabled = true;
    toast(apiErrorMessage(error), true);
  } finally {
    button.disabled = false;
    button.textContent = "Run preflight";
  }
}

async function startRun() {
  if (
    !state.preflightReady
    || state.preflightRevision !== state.formRevision
    || !state.preflightPayload
    || !validateStoryDirection()
  ) return;
  const button = byId("start-run-button");
  const payload = state.preflightPayload;
  button.disabled = true;
  button.textContent = "Creating…";
  try {
    const created = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
    byId("new-run-dialog").close();
    toast("Run created and queued.");
    await refreshRuns();
    await selectRun(created.run_id);
  } catch (error) {
    toast(apiErrorMessage(error), true);
    button.disabled = false;
  } finally {
    button.textContent = "Start run";
  }
}

function openNewRunDialog() {
  invalidatePreflight();
  byId("form-status").textContent = "Preflight is local and read-only. It will not call paid cloud models.";
  byId("new-run-dialog").showModal();
  window.setTimeout(() => byId("idea-direction").focus(), 40);
}

async function resumeSelected() {
  if (!state.selectedRunId) return;
  try {
    await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/resume`, { method: "POST", body: "{}" });
    toast("Run queued for resume.");
    await refreshDetail({ silent: true });
    connectRunEvents(state.selectedRunId);
  } catch (error) {
    toast(apiErrorMessage(error), true);
  }
}

async function stopSelected() {
  if (!state.selectedRunId) return;
  try {
    const result = await api(`/api/runs/${encodeURIComponent(state.selectedRunId)}/stop`, { method: "POST", body: "{}" });
    toast(result.warning || "Stop requested. An accepted cloud request may still be billed.", Boolean(result.warning));
    await refreshDetail({ silent: true });
  } catch (error) {
    toast(apiErrorMessage(error), true);
  }
}

function bindEvents() {
  byId("new-run-button").addEventListener("click", openNewRunDialog);
  byId("empty-new-run").addEventListener("click", openNewRunDialog);
  byId("dialog-close").addEventListener("click", () => byId("new-run-dialog").close());
  byId("new-run-form").addEventListener("submit", (event) => event.preventDefault());
  byId("preflight-button").addEventListener("click", runPreflight);
  byId("start-run-button").addEventListener("click", startRun);
  byId("resume-button").addEventListener("click", resumeSelected);
  byId("stop-button").addEventListener("click", stopSelected);
  byId("refresh-button").addEventListener("click", () => refreshDetail());
  byId("run-filter").addEventListener("input", renderRuns);
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    state.activeTab = tab.dataset.tab;
    renderActiveTab();
  }));
  byId("profile").addEventListener("change", () => { applyProfileModels(); invalidatePreflight(); });
  byId("orientation").addEventListener("change", invalidatePreflight);
  byId("surprise-me").addEventListener("change", () => {
    byId("idea-direction").disabled = byId("surprise-me").checked;
    byId("idea-direction").setCustomValidity("");
    invalidatePreflight();
  });
  byId("animated-captions").addEventListener("change", () => {
    if (byId("animated-captions").checked) byId("captions-enabled").checked = true;
    invalidatePreflight();
  });
  byId("new-run-form").addEventListener("input", (event) => {
    if (!event.target.matches("[data-model-task]")) invalidatePreflight();
  });
  byId("new-run-dialog").addEventListener("cancel", () => byId("new-run-dialog").close());
}

async function initialize() {
  try {
    const bootstrap = await api("/api/bootstrap");
    state.bootstrap = bootstrap;
    state.token = bootstrap.dashboard_token;
    state.runs = bootstrap.runs;
    byId("app-version").textContent = `v${bootstrap.version}`;
    applyDefaults();
    bindEvents();
    renderRuns();
    if (state.runs.length) await selectRun(state.runs[0].run_id);
  } catch (error) {
    byId("empty-state").querySelector("h1").textContent = "The dashboard could not initialize.";
    byId("empty-state").querySelector("p:not(.eyebrow)").textContent = apiErrorMessage(error);
    byId("empty-new-run").classList.add("is-hidden");
  }
}

initialize();
