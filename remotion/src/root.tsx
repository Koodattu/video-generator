import React from "react";
import {Composition} from "remotion";

import {ExplainerVideo} from "./video";
import type {RenderManifest} from "./types";

const placeholder: RenderManifest = {
  schemaVersion: 1,
  title: "Remotion explainer",
  width: 1280,
  height: 720,
  fps: 30,
  durationFrames: 90,
  assetBaseUrl: "",
  labels: {
    payAttention: "PAY ATTENTION",
    source: "THE SOURCE",
    citedSource: "cited source",
    takeaway: "THE TAKEAWAY",
    before: "Before",
    after: "After"
  },
  narrationFile: "",
  captionsEnabled: true,
  captionWords: [
    {wordId: "word-000001", text: "A", startFrame: 0, endFrame: 10},
    {wordId: "word-000002", text: "fast", startFrame: 10, endFrame: 25},
    {wordId: "word-000003", text: "local", startFrame: 25, endFrame: 45},
    {wordId: "word-000004", text: "video", startFrame: 45, endFrame: 65},
    {wordId: "word-000005", text: "renderer.", startFrame: 65, endFrame: 90}
  ],
  sfxFiles: {},
  shots: [
    {
      shotId: "shot-001",
      sceneId: "scene-001",
      startWordId: "word-000001",
      endWordId: "word-000001",
      narrationExcerpt: "A fast local video renderer.",
      durationFrames: 90,
      template: "kinetic_hook",
      purpose: "Preview the fixed template library.",
      headline: "LOCAL. FAST. REPEATABLE.",
      supportingText: "Narration-locked Remotion templates",
      bodyLines: [],
      motion: "punch_in",
      transitionIn: "hard_cut",
      sfx: "none"
    }
  ]
};

export const VideoRoot: React.FC = () => {
  return (
    <Composition
      id="LocalExplainer"
      component={ExplainerVideo}
      durationInFrames={placeholder.durationFrames}
      fps={placeholder.fps}
      width={placeholder.width}
      height={placeholder.height}
      defaultProps={placeholder}
      calculateMetadata={({props}) => ({
        durationInFrames: props.durationFrames,
        fps: props.fps,
        width: props.width,
        height: props.height
      })}
    />
  );
};
