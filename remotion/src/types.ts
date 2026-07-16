export type TemplateName =
  | "kinetic_hook"
  | "headline_zoom"
  | "source_screenshot"
  | "code_reveal"
  | "diagram_flow"
  | "comparison_split"
  | "meme_cutaway"
  | "conclusion";

export type MotionPreset =
  | "punch_in"
  | "slide_up"
  | "pan"
  | "type_on"
  | "build"
  | "hold";

export type SfxPreset = "none" | "click" | "pop" | "whoosh";

export type RenderShot = {
  shotId: string;
  sceneId: string;
  startWordId: string;
  endWordId: string;
  narrationExcerpt: string;
  durationFrames: number;
  template: TemplateName;
  purpose: string;
  headline: string;
  supportingText: string;
  bodyLines: string[];
  motion: MotionPreset;
  transitionIn: "hard_cut" | "section_wipe";
  sfx: SfxPreset;
  assetFile?: string;
  assetMediaKind?: "image" | "video";
  sourceLabel?: string;
};

export type CaptionWord = {
  wordId: string;
  text: string;
  startFrame: number;
  endFrame: number;
};

export type RenderLabels = {
  payAttention: string;
  source: string;
  citedSource: string;
  takeaway: string;
  before: string;
  after: string;
};

export type RenderManifest = {
  schemaVersion: 1;
  title: string;
  width: number;
  height: number;
  fps: number;
  durationFrames: number;
  assetBaseUrl: string;
  labels: RenderLabels;
  narrationFile: string;
  captionsEnabled: boolean;
  captionWords: CaptionWord[];
  musicFile?: string;
  sfxFiles: Partial<Record<Exclude<SfxPreset, "none">, string>>;
  shots: RenderShot[];
};
