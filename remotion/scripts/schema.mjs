import {z} from "zod";

const shot = z.object({
  shotId: z.string().regex(/^shot-\d{3,}$/),
  sceneId: z.string().regex(/^scene-\d{3,}$/),
  startWordId: z.string().regex(/^word-\d{6,}$/),
  endWordId: z.string().regex(/^word-\d{6,}$/),
  narrationExcerpt: z.string().min(1).max(4000),
  durationFrames: z.number().int().positive(),
  template: z.enum(["kinetic_hook", "headline_zoom", "source_screenshot", "code_reveal", "diagram_flow", "comparison_split", "meme_cutaway", "conclusion"]),
  purpose: z.string().min(1).max(240),
  headline: z.string().min(1).max(80),
  supportingText: z.string().max(160),
  bodyLines: z.array(z.string().min(1).max(120)).max(8),
  motion: z.enum(["punch_in", "slide_up", "pan", "type_on", "build", "hold"]),
  transitionIn: z.enum(["hard_cut", "section_wipe"]),
  sfx: z.enum(["none", "click", "pop", "whoosh"]),
  assetFile: z.string().min(1).optional(),
  assetMediaKind: z.enum(["image", "video"]).optional(),
  sourceLabel: z.string().max(300).optional()
}).strict().superRefine((value, context) => {
  if ((value.assetFile === undefined) !== (value.assetMediaKind === undefined)) {
    context.addIssue({code: "custom", message: "assetFile and assetMediaKind must appear together"});
  }
  if (value.template === "source_screenshot" && !value.assetFile) {
    context.addIssue({code: "custom", message: "source_screenshot requires a local asset"});
  }
  if (value.template === "code_reveal" && value.bodyLines.length < 2) {
    context.addIssue({code: "custom", message: "code_reveal requires at least two body lines"});
  }
  if (value.template === "diagram_flow" && value.bodyLines.length < 2) {
    context.addIssue({code: "custom", message: "diagram_flow requires at least two body lines"});
  }
  if (value.template === "comparison_split" && value.bodyLines.length !== 2) {
    context.addIssue({code: "custom", message: "comparison_split requires exactly two body lines"});
  }
  if (value.template === "diagram_flow" && (value.bodyLines.length > 5 || value.bodyLines.some((line) => line.length > 32))) {
    context.addIssue({code: "custom", message: "diagram_flow body line budget exceeded"});
  }
  if (value.template === "code_reveal" && value.bodyLines.some((line) => line.length > 70)) {
    context.addIssue({code: "custom", message: "code_reveal body line budget exceeded"});
  }
  if (value.template === "comparison_split" && value.bodyLines.some((line) => line.length > 60)) {
    context.addIssue({code: "custom", message: "comparison_split body line budget exceeded"});
  }
  if (!["code_reveal", "diagram_flow", "comparison_split"].includes(value.template) && value.bodyLines.length) {
    context.addIssue({code: "custom", message: `${value.template} does not render bodyLines`});
  }
  if (!["kinetic_hook", "headline_zoom", "source_screenshot", "meme_cutaway"].includes(value.template) && value.assetFile) {
    context.addIssue({code: "custom", message: `${value.template} does not render an asset`});
  }
  if (value.template === "code_reveal" && value.motion !== "type_on") {
    context.addIssue({code: "custom", message: "code_reveal requires type_on motion"});
  }
  if (value.template === "diagram_flow" && value.motion !== "build") {
    context.addIssue({code: "custom", message: "diagram_flow requires build motion"});
  }
});

const captionWord = z.object({
  wordId: z.string().regex(/^word-\d{6,}$/),
  text: z.string().min(1).max(200),
  startFrame: z.number().int().nonnegative(),
  endFrame: z.number().int().positive()
}).strict().superRefine((value, context) => {
  if (value.endFrame <= value.startFrame) {
    context.addIssue({code: "custom", message: "caption word endFrame must follow startFrame"});
  }
});

export const renderManifestSchema = z.object({
  schemaVersion: z.literal(1),
  title: z.string().min(1).max(200),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  fps: z.number().int().positive(),
  durationFrames: z.number().int().positive(),
  assetBaseUrl: z.string(),
  labels: z.object({
    payAttention: z.string().min(1).max(80),
    source: z.string().min(1).max(80),
    citedSource: z.string().min(1).max(80),
    takeaway: z.string().min(1).max(80),
    before: z.string().min(1).max(80),
    after: z.string().min(1).max(80)
  }).strict(),
  narrationFile: z.string().min(1),
  captionsEnabled: z.boolean(),
  captionWords: z.array(captionWord).min(1),
  musicFile: z.string().min(1).optional(),
  sfxFiles: z.object({
    click: z.string().min(1).optional(),
    pop: z.string().min(1).optional(),
    whoosh: z.string().min(1).optional()
  }).strict(),
  shots: z.array(shot).min(1)
}).strict().superRefine((value, context) => {
  const total = value.shots.reduce((sum, item) => sum + item.durationFrames, 0);
  if (total !== value.durationFrames) {
    context.addIssue({code: "custom", message: "Shot frames must sum to durationFrames"});
  }
  const expectedWordIds = value.captionWords.map((_, index) => `word-${String(index + 1).padStart(6, "0")}`);
  if (value.captionWords.some((word, index) => word.wordId !== expectedWordIds[index])) {
    context.addIssue({code: "custom", message: "caption word IDs must be contiguous and ordered"});
  }
  if (value.captionWords.some((word) => word.endFrame > value.durationFrames)) {
    context.addIssue({code: "custom", message: "caption word timing exceeds durationFrames"});
  }
  if (value.captionWords.some((word, index) => index > 0 && word.startFrame < value.captionWords[index - 1].startFrame)) {
    context.addIssue({code: "custom", message: "caption words must be time-ordered"});
  }
  const sectionTransitions = value.shots.filter((shot) => shot.transitionIn === "section_wipe");
  if (value.shots[0]?.transitionIn !== "hard_cut") {
    context.addIssue({code: "custom", message: "the first Shot must use a hard cut"});
  }
  if (value.shots.length > 1 && sectionTransitions.length !== 1) {
    context.addIssue({code: "custom", message: "multi-Shot manifests require one section_wipe"});
  }
});
