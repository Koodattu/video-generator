import React from "react";
import {Audio, Video} from "@remotion/media";
import {
  AbsoluteFill,
  Img,
  Series,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig
} from "remotion";

import type {CaptionWord, RenderLabels, RenderManifest, RenderShot} from "./types";

const colors = {
  ink: "#f7f7f2",
  paper: "#101114",
  panel: "#191b21",
  muted: "#a8acb8",
  yellow: "#ffd43b",
  cyan: "#47d7ff",
  coral: "#ff5d73",
  green: "#6ee7a8"
};

const joinAsset = (base: string, file: string): string => {
  const prefix = base.endsWith("/") ? base : `${base}/`;
  return `${prefix}${file.split("/").map(encodeURIComponent).join("/")}`;
};

const useEntrance = (motion: RenderShot["motion"]) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const progress = spring({frame, fps, config: {damping: 16, stiffness: 180}});
  if (motion === "hold") return {opacity: 1, transform: "none"};
  if (motion === "slide_up") {
    return {
      opacity: interpolate(frame, [0, 6], [0, 1], {extrapolateRight: "clamp"}),
      transform: `translateY(${interpolate(progress, [0, 1], [90, 0])}px)`
    };
  }
  if (motion === "pan") {
    return {opacity: 1, transform: `translateX(${interpolate(frame, [0, fps * 2], [-25, 0], {extrapolateRight: "clamp"})}px)`};
  }
  return {
    opacity: interpolate(frame, [0, 4], [0, 1], {extrapolateRight: "clamp"}),
    transform: `scale(${interpolate(progress, [0, 1], [motion === "punch_in" ? 1.22 : 0.92, 1])})`
  };
};

const usePortrait = (): boolean => {
  const {width, height} = useVideoConfig();
  return height > width;
};

const MediaLayer: React.FC<{shot: RenderShot; base: string; dim?: boolean}> = ({shot, base, dim = false}) => {
  if (!shot.assetFile || !shot.assetMediaKind) return null;
  const src = joinAsset(base, shot.assetFile);
  const common: React.CSSProperties = {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    filter: dim ? "brightness(0.42) saturate(0.8)" : "none"
  };
  return shot.assetMediaKind === "video" ? (
    <Video src={src} loop muted style={common} />
  ) : (
    <Img src={src} style={common} />
  );
};

const Headline: React.FC<{shot: RenderShot; compact?: boolean}> = ({shot, compact = false}) => {
  const portrait = usePortrait();
  return (
    <div style={{maxWidth: portrait ? "100%" : compact ? 900 : 1120}}>
      <div
        style={{
          color: colors.ink,
          fontFamily: "Arial Black, Arial, sans-serif",
          fontSize: portrait ? (compact ? 58 : 72) : compact ? 64 : 92,
          fontWeight: 900,
          letterSpacing: portrait ? -2 : -3,
          lineHeight: 0.95,
          overflowWrap: "anywhere",
          textTransform: "uppercase",
          textWrap: "balance"
        }}
      >
        {shot.headline}
      </div>
      {shot.supportingText ? (
        <div style={{color: colors.muted, fontFamily: "Arial, sans-serif", fontSize: portrait ? 26 : 30, fontWeight: 700, lineHeight: 1.2, marginTop: portrait ? 20 : 26}}>
          {shot.supportingText}
        </div>
      ) : null}
    </div>
  );
};

const KineticHook: React.FC<{shot: RenderShot; base: string; labels: RenderLabels}> = ({shot, base, labels}) => {
  const frame = useCurrentFrame();
  const entrance = useEntrance(shot.motion);
  const portrait = usePortrait();
  const accent = frame % 14 < 7 ? colors.yellow : colors.coral;
  return (
    <AbsoluteFill style={{backgroundColor: colors.paper, overflow: "hidden"}}>
      <MediaLayer shot={shot} base={base} dim />
      <div style={{position: "absolute", inset: 0, background: portrait ? "linear-gradient(0deg, rgba(16,17,20,.96), rgba(16,17,20,.2))" : "linear-gradient(90deg, rgba(16,17,20,.95), rgba(16,17,20,.2))"}} />
      <div style={{...entrance, position: "absolute", left: portrait ? 46 : 72, right: portrait ? 46 : 72, top: portrait ? "16%" : 110}}>
        <div style={{background: accent, color: colors.paper, display: "inline-block", fontFamily: "Arial Black, Arial", fontSize: 22, fontWeight: 900, letterSpacing: 3, marginBottom: 24, padding: "10px 16px"}}>
          {labels.payAttention}
        </div>
        <Headline shot={shot} />
      </div>
    </AbsoluteFill>
  );
};

const HeadlineZoom: React.FC<{shot: RenderShot; base: string}> = ({shot, base}) => {
  const entrance = useEntrance(shot.motion);
  const portrait = usePortrait();
  return (
    <AbsoluteFill style={{background: `radial-gradient(circle at 75% 30%, #243041, ${colors.paper} 58%)`}}>
      <div style={{position: "absolute", right: 0, top: 0, width: portrait ? "100%" : "48%", height: portrait ? "54%" : "100%", opacity: 0.8}}>
        <MediaLayer shot={shot} base={base} />
      </div>
      <div style={{...entrance, alignItems: "flex-start", bottom: portrait ? 0 : "auto", boxSizing: "border-box", display: "flex", height: portrait ? "50%" : "100%", justifyContent: "center", flexDirection: "column", padding: portrait ? "36px 46px" : "0 70px", position: portrait ? "absolute" : "relative", width: portrait ? "100%" : "65%"}}>
        <Headline shot={shot} compact />
      </div>
    </AbsoluteFill>
  );
};

const SourceScreenshot: React.FC<{shot: RenderShot; base: string; labels: RenderLabels}> = ({shot, base, labels}) => {
  const entrance = useEntrance(shot.motion);
  const portrait = usePortrait();
  return (
    <AbsoluteFill style={{background: colors.paper, boxSizing: "border-box", display: "flex", flexDirection: "column", padding: portrait ? 36 : 48}}>
      <div style={{color: colors.yellow, fontFamily: "Arial Black, Arial", fontSize: 22, letterSpacing: 2, marginBottom: 18}}>{labels.source}</div>
      <div style={{background: "white", border: `8px solid ${colors.panel}`, borderRadius: 18, boxShadow: "0 20px 80px rgba(0,0,0,.55)", flex: 1, minHeight: 0, overflow: "hidden", position: "relative", ...entrance}}>
        <MediaLayer shot={shot} base={base} />
      </div>
      <div style={{alignItems: portrait ? "flex-start" : "center", display: "flex", flexDirection: portrait ? "column" : "row", gap: portrait ? 8 : 0, justifyContent: "space-between", marginTop: 18}}>
        <div style={{color: colors.ink, fontFamily: "Arial Black, Arial", fontSize: portrait ? 28 : 34, fontWeight: 900}}>{shot.headline}</div>
        <div style={{color: colors.muted, fontFamily: "Arial", fontSize: 18}}>{shot.sourceLabel || labels.citedSource}</div>
      </div>
    </AbsoluteFill>
  );
};

const CodeReveal: React.FC<{shot: RenderShot}> = ({shot}) => {
  const frame = useCurrentFrame();
  const portrait = usePortrait();
  return (
    <AbsoluteFill style={{background: colors.paper, boxSizing: "border-box", padding: portrait ? 36 : 58}}>
      <div style={{color: colors.ink, fontFamily: "Arial Black, Arial", fontSize: portrait ? 42 : 50, fontWeight: 900, marginBottom: 28}}>{shot.headline}</div>
      <div style={{background: "#0b0c0f", border: "2px solid #30333d", borderRadius: 18, boxShadow: "0 24px 90px rgba(0,0,0,.5)", flex: 1, overflow: "hidden"}}>
        <div style={{background: "#20222a", display: "flex", gap: 10, padding: 16}}>
          {[colors.coral, colors.yellow, colors.green].map((color) => <div key={color} style={{background: color, borderRadius: 99, height: 14, width: 14}} />)}
        </div>
        <div style={{fontFamily: "Consolas, monospace", fontSize: portrait ? 22 : 31, lineHeight: 1.55, overflowWrap: "anywhere", padding: portrait ? 24 : 34}}>
          {shot.bodyLines.map((line, index) => {
            const visible = frame >= index * 7;
            return (
              <div key={`${index}-${line}`} style={{color: index % 2 ? colors.cyan : colors.green, opacity: visible ? 1 : 0, transform: `translateX(${visible ? 0 : -20}px)`}}>
                <span style={{color: "#656a78", display: "inline-block", marginRight: 28, textAlign: "right", width: 40}}>{index + 1}</span>{line}
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};

const DiagramFlow: React.FC<{shot: RenderShot}> = ({shot}) => {
  const frame = useCurrentFrame();
  const portrait = usePortrait();
  const lastActivationFrame = Math.max(1, shot.durationFrames - 8);
  return (
    <AbsoluteFill style={{background: colors.paper, boxSizing: "border-box", padding: portrait ? "44px 34px" : "70px 54px"}}>
      <div style={{color: colors.ink, fontFamily: "Arial Black, Arial", fontSize: portrait ? 42 : 52, fontWeight: 900, textAlign: "center"}}>{shot.headline}</div>
      <div style={{alignItems: "center", display: "flex", flex: 1, flexDirection: portrait ? "column" : "row", justifyContent: "center", marginTop: 30, minWidth: 0}}>
        {shot.bodyLines.map((line, index) => {
          const activationFrame = shot.bodyLines.length === 1
            ? 0
            : Math.round((index / (shot.bodyLines.length - 1)) * lastActivationFrame);
          const active = frame >= activationFrame;
          const entrance = interpolate(frame, [activationFrame, activationFrame + 5], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp"
          });
          return (
            <React.Fragment key={`${index}-${line}`}>
              {index ? <div style={{color: colors.yellow, flex: portrait ? "0 1 30px" : "0 1 44px", fontFamily: "Arial Black", fontSize: portrait ? 32 : 42, opacity: entrance, textAlign: "center", transform: portrait ? `scaleY(${entrance})` : `scaleX(${entrance})`, transformOrigin: portrait ? "center top" : "left center"}}>{portrait ? "↓" : "→"}</div> : null}
              <div style={{background: index % 2 ? colors.cyan : colors.yellow, borderRadius: 18, color: colors.paper, flex: portrait ? "0 1 auto" : "1 1 0", fontFamily: "Arial Black, Arial", fontSize: 24, fontWeight: 900, maxWidth: portrait ? "100%" : 230, minWidth: 0, opacity: active ? 1 : 0.16, padding: portrait ? "24px 28px" : "30px 16px", textAlign: "center", transform: `translateY(${(1 - entrance) * 22}px) scale(${0.9 + entrance * 0.1})`, width: portrait ? "82%" : "auto"}}>{line}</div>
            </React.Fragment>
          );
        })}
      </div>
      <div style={{color: colors.muted, fontFamily: "Arial", fontSize: 25, textAlign: "center"}}>{shot.supportingText}</div>
    </AbsoluteFill>
  );
};

const ComparisonSplit: React.FC<{shot: RenderShot; labels: RenderLabels}> = ({shot, labels}) => {
  const entrance = useEntrance(shot.motion);
  const portrait = usePortrait();
  return (
    <AbsoluteFill style={{background: colors.paper, boxSizing: "border-box", padding: portrait ? 36 : 48}}>
      <div style={{color: colors.ink, fontFamily: "Arial Black, Arial", fontSize: portrait ? 40 : 48, fontWeight: 900, textAlign: "center"}}>{shot.headline}</div>
      <div style={{display: "grid", flex: 1, gap: 24, gridTemplateColumns: portrait ? "1fr" : "1fr 1fr", gridTemplateRows: portrait ? "1fr 1fr" : "none", marginTop: 34, ...entrance}}>
        {(shot.bodyLines.length === 2 ? shot.bodyLines : [labels.before, labels.after]).map((line, index) => (
          <div key={`${index}-${line}`} style={{alignItems: "center", background: index ? colors.cyan : colors.coral, borderRadius: 24, color: colors.paper, display: "flex", fontFamily: "Arial Black, Arial", fontSize: portrait ? 36 : 43, fontWeight: 900, justifyContent: "center", padding: portrait ? 30 : 42, textAlign: "center"}}>{line}</div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

const MemeCutaway: React.FC<{shot: RenderShot; base: string}> = ({shot, base}) => {
  const entrance = useEntrance(shot.motion);
  const portrait = usePortrait();
  return (
    <AbsoluteFill style={{background: colors.paper}}>
      <MediaLayer shot={shot} base={base} dim />
      <div style={{position: "absolute", inset: 0, background: "linear-gradient(0deg, rgba(16,17,20,.95), rgba(16,17,20,.05) 70%)"}} />
      <div style={{...entrance, bottom: portrait ? 70 : 48, left: portrait ? 40 : 55, position: "absolute", right: portrait ? 40 : 55}}>
        <div style={{color: colors.ink, fontFamily: "Arial Black, Arial", fontSize: portrait ? 48 : 58, fontWeight: 900, lineHeight: 1, textShadow: "0 4px 18px black", textTransform: "uppercase"}}>{shot.headline}</div>
        <div style={{color: colors.yellow, fontFamily: "Arial", fontSize: portrait ? 24 : 27, fontWeight: 800, marginTop: 18}}>{shot.supportingText}</div>
      </div>
    </AbsoluteFill>
  );
};

const Conclusion: React.FC<{shot: RenderShot; labels: RenderLabels}> = ({shot, labels}) => {
  const entrance = useEntrance(shot.motion);
  const portrait = usePortrait();
  return (
    <AbsoluteFill style={{alignItems: "center", background: `linear-gradient(135deg, ${colors.paper}, #182836)`, boxSizing: "border-box", display: "flex", justifyContent: "center", padding: portrait ? 46 : 70, textAlign: "center"}}>
      <div style={entrance}>
        <div style={{color: colors.yellow, fontFamily: "Arial Black, Arial", fontSize: 22, letterSpacing: 4, marginBottom: 26}}>{labels.takeaway}</div>
        <Headline shot={shot} compact />
      </div>
    </AbsoluteFill>
  );
};

const ShotTemplate: React.FC<{shot: RenderShot; base: string; labels: RenderLabels}> = ({shot, base, labels}) => {
  switch (shot.template) {
    case "kinetic_hook": return <KineticHook shot={shot} base={base} labels={labels} />;
    case "headline_zoom": return <HeadlineZoom shot={shot} base={base} />;
    case "source_screenshot": return <SourceScreenshot shot={shot} base={base} labels={labels} />;
    case "code_reveal": return <CodeReveal shot={shot} />;
    case "diagram_flow": return <DiagramFlow shot={shot} />;
    case "comparison_split": return <ComparisonSplit shot={shot} labels={labels} />;
    case "meme_cutaway": return <MemeCutaway shot={shot} base={base} />;
    case "conclusion": return <Conclusion shot={shot} labels={labels} />;
  }
};

const SectionWipe: React.FC = () => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [0, 3, 8], [-110, 0, 110], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp"
  });
  if (frame > 8) return null;
  return (
    <AbsoluteFill
      style={{
        background: `linear-gradient(90deg, ${colors.yellow} 0 72%, ${colors.coral} 72% 84%, ${colors.paper} 84%)`,
        transform: `translateX(${progress}%)`,
        zIndex: 40
      }}
    />
  );
};

const captionPages = (words: CaptionWord[]): CaptionWord[][] => {
  const pages: CaptionWord[][] = [];
  let page: CaptionWord[] = [];
  let characterUnits = 0;
  for (const word of words) {
    const units = [...word.text].length + (page.length ? 1 : 0);
    if (page.length && (page.length >= 5 || characterUnits + units > 38)) {
      pages.push(page);
      page = [];
      characterUnits = 0;
    }
    page.push(word);
    characterUnits += [...word.text].length + (page.length > 1 ? 1 : 0);
  }
  if (page.length) pages.push(page);
  return pages;
};

const KineticCaptions: React.FC<{words: CaptionWord[]; width: number}> = ({words, width}) => {
  const frame = useCurrentFrame();
  let activeIndex = -1;
  for (let index = 0; index < words.length; index += 1) {
    if (words[index].startFrame > frame) break;
    activeIndex = index;
  }
  if (activeIndex < 0) return null;
  const activeWord = words[activeIndex];
  if (activeIndex === words.length - 1 && frame >= activeWord.endFrame) return null;
  const page = captionPages(words).find((candidate) => candidate.includes(activeWord)) ?? [activeWord];
  const defaultFontSize = Math.max(32, Math.round(width * 0.029));
  const longestWord = Math.max(...page.map((word) => [...word.text].length));
  const fontSize = Math.max(
    Math.round(width * 0.019),
    Math.round(defaultFontSize * Math.min(1, 28 / Math.max(1, longestWord)))
  );
  return (
    <div
      style={{
        alignItems: "center",
        bottom: Math.max(18, Math.round(width * 0.014)),
        display: "flex",
        justifyContent: "center",
        left: 0,
        pointerEvents: "none",
        position: "absolute",
        right: 0,
        zIndex: 50
      }}
    >
      <div
        style={{
          background: "rgba(8, 9, 12, 0.88)",
          border: "2px solid rgba(255,255,255,.16)",
          borderRadius: 16,
          boxShadow: "0 10px 35px rgba(0,0,0,.45)",
          color: colors.ink,
          display: "flex",
          flexWrap: "wrap",
          fontFamily: "Arial Black, Arial, sans-serif",
          fontSize,
          fontWeight: 900,
          columnGap: Math.max(16, Math.round(width * 0.0125)),
          justifyContent: "center",
          letterSpacing: -1,
          lineHeight: 1.05,
          maxWidth: "88%",
          padding: `${Math.max(10, Math.round(width * 0.008))}px ${Math.max(18, Math.round(width * 0.014))}px`,
          textAlign: "center",
          textShadow: "0 2px 8px rgba(0,0,0,.65)",
          textWrap: "pretty",
          overflowWrap: "anywhere",
          wordBreak: "break-word"
        }}
      >
        {page.map((word) => {
          const active = word.wordId === activeWord.wordId && frame < word.endFrame;
          return (
            <span
              key={word.wordId}
              style={{
                color: active ? colors.yellow : colors.ink,
                display: "inline-block",
                maxWidth: "100%",
                opacity: active ? 1 : 0.78,
                transform: `scale(${active ? 1.04 : 1})`,
                transformOrigin: "center center"
              }}
            >
              {word.text}
            </span>
          );
        })}
      </div>
    </div>
  );
};

export const ExplainerVideo: React.FC<RenderManifest> = (manifest) => {
  const captionSafeArea = manifest.captionsEnabled
    ? Math.max(168, Math.round(manifest.width * 0.14))
    : 0;
  return (
    <AbsoluteFill style={{backgroundColor: colors.paper}}>
      {manifest.narrationFile ? <Audio src={joinAsset(manifest.assetBaseUrl, manifest.narrationFile)} volume={1} /> : null}
      {manifest.musicFile ? <Audio src={joinAsset(manifest.assetBaseUrl, manifest.musicFile)} volume={0.11} /> : null}
      <Series>
        {manifest.shots.map((shot) => {
          const sfxFile = shot.sfx === "none" ? undefined : manifest.sfxFiles[shot.sfx];
          return (
            <Series.Sequence key={shot.shotId} durationInFrames={shot.durationFrames}>
              <AbsoluteFill
                style={{
                  bottom: "auto",
                  boxSizing: "border-box",
                  height: `calc(100% - ${captionSafeArea}px)`,
                  overflow: "hidden"
                }}
              >
                <ShotTemplate shot={shot} base={manifest.assetBaseUrl} labels={manifest.labels} />
              </AbsoluteFill>
              {shot.transitionIn === "section_wipe" ? <SectionWipe /> : null}
              {sfxFile ? <Audio src={joinAsset(manifest.assetBaseUrl, sfxFile)} volume={0.16} /> : null}
            </Series.Sequence>
          );
        })}
      </Series>
      {manifest.captionsEnabled ? (
        <KineticCaptions words={manifest.captionWords} width={manifest.width} />
      ) : null}
    </AbsoluteFill>
  );
};
