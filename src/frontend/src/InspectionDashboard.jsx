/**
 * InspectionDashboard.jsx
 * Concrete Structural Damage Inspection System — Phase 4, Stage 2
 *
 * Field-facing dashboard: upload a structural image, see what the models found
 * drawn over it, read the severity assessment, download the engineering report.
 *
 * Design position: this is a site record sheet, not a product dashboard.
 * Everything is neutral concrete grey except the severity grade, which is the
 * one thing an engineer scanning fifty results needs to read at a glance.
 * Measurements use tabular figures so columns of numbers line up.
 *
 * Requires Tailwind CSS. No external UI library.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env?.VITE_API_BASE || "http://localhost:8000";

/* ---------------------------------------------------------------------------
 * Grade presentation. Colours are deliberately desaturated relative to typical
 * status palettes — these sit against concrete photography, and fully saturated
 * chips vibrate against grey aggregate.
 * ------------------------------------------------------------------------- */
const GRADES = {
  "No damage detected": { bg: "bg-[#4E8C57]", text: "text-[#4E8C57]", border: "border-[#4E8C57]" },
  Low: { bg: "bg-[#3D7EA6]", text: "text-[#3D7EA6]", border: "border-[#3D7EA6]" },
  Medium: { bg: "bg-[#C4901F]", text: "text-[#C4901F]", border: "border-[#C4901F]" },
  High: { bg: "bg-[#C4601F]", text: "text-[#C4601F]", border: "border-[#C4601F]" },
  Critical: { bg: "bg-[#A62B2B]", text: "text-[#A62B2B]", border: "border-[#A62B2B]" },
};
const gradeStyle = (g) => GRADES[g] || { bg: "bg-stone-500", text: "text-stone-600", border: "border-stone-400" };

const BOX_COLOUR = "#E8B33C";
const MASK_TINT = [214, 62, 62];

/* ---------------------------------------------------------------------------
 * Small presentational pieces
 * ------------------------------------------------------------------------- */
function Figure({ label, value, unit, muted }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-stone-200 py-2 last:border-0">
      <span className="text-[13px] leading-snug text-stone-600">{label}</span>
      <span
        className={`text-[13px] tabular-nums ${
          muted ? "text-stone-400" : "font-medium text-stone-900"
        }`}
      >
        {value}
        {unit && <span className="ml-1 text-stone-500">{unit}</span>}
      </span>
    </div>
  );
}

function SectionHeading({ children, note }) {
  return (
    <div className="mb-3 mt-7 flex items-baseline justify-between first:mt-0">
      <h2 className="text-[15px] font-semibold tracking-tight text-stone-900">{children}</h2>
      {note && <span className="text-[11px] text-stone-500">{note}</span>}
    </div>
  );
}

function Toggle({ checked, onChange, children }) {
  return (
    <label className="flex cursor-pointer select-none items-center gap-2 text-[13px] text-stone-700">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-4 w-4 rounded border-stone-400 text-stone-900 focus:ring-2 focus:ring-stone-900 focus:ring-offset-1"
      />
      {children}
    </label>
  );
}

/* ---------------------------------------------------------------------------
 * Canvas: source image + optional mask tint + optional detection boxes.
 * Drawn at natural resolution then CSS-scaled, so boxes stay pixel-accurate
 * regardless of the display size.
 * ------------------------------------------------------------------------- */
function AnnotatedCanvas({ imageUrl, maskBase64, detections, showMask, showBoxes, maskOpacity, onHoverBox }) {
  const canvasRef = useRef(null);
  const [dims, setDims] = useState({ w: 0, h: 0 });
  const boxesRef = useRef([]);

  useEffect(() => {
    if (!imageUrl) return;
    let cancelled = false;

    const load = (src) =>
      new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
      });

    (async () => {
      try {
        const img = await load(imageUrl);
        if (cancelled) return;

        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        setDims({ w: img.naturalWidth, h: img.naturalHeight });

        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        if (showMask && maskBase64) {
          const mask = await load(`data:image/png;base64,${maskBase64}`);
          if (cancelled) return;

          const off = document.createElement("canvas");
          off.width = canvas.width;
          off.height = canvas.height;
          const octx = off.getContext("2d");
          octx.drawImage(mask, 0, 0, canvas.width, canvas.height);

          const md = octx.getImageData(0, 0, canvas.width, canvas.height);
          const base = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const a = maskOpacity;
          for (let i = 0; i < md.data.length; i += 4) {
            if (md.data[i] > 127) {
              base.data[i] = base.data[i] * (1 - a) + MASK_TINT[0] * a;
              base.data[i + 1] = base.data[i + 1] * (1 - a) + MASK_TINT[1] * a;
              base.data[i + 2] = base.data[i + 2] * (1 - a) + MASK_TINT[2] * a;
            }
          }
          ctx.putImageData(base, 0, 0);
        }

        if (showBoxes && detections?.length) {
          const lw = Math.max(2, Math.round(canvas.width / 500));
          const fs = Math.max(12, Math.round(canvas.width / 60));
          ctx.lineWidth = lw;
          ctx.strokeStyle = BOX_COLOUR;
          ctx.font = `600 ${fs}px ui-sans-serif, system-ui, sans-serif`;
          ctx.textBaseline = "bottom";

          detections.forEach((d) => {
            const [x1, y1, x2, y2] = d.bbox_xyxy;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = BOX_COLOUR;
            ctx.fillRect(x1, Math.max(0, y1 - fs - 6), tw + 10, fs + 6);
            ctx.fillStyle = "#1c1917";
            ctx.fillText(label, x1 + 5, Math.max(fs, y1 - 3));
          });
          boxesRef.current = detections;
        }
      } catch {
        /* image failed to decode; the empty canvas is the visible signal */
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [imageUrl, maskBase64, detections, showMask, showBoxes, maskOpacity]);

  const handleMove = useCallback(
    (e) => {
      if (!onHoverBox || !boxesRef.current.length) return;
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width;
      const sy = canvas.height / rect.height;
      const x = (e.clientX - rect.left) * sx;
      const y = (e.clientY - rect.top) * sy;
      const hit = boxesRef.current.find(
        (d) => x >= d.bbox_xyxy[0] && x <= d.bbox_xyxy[2] && y >= d.bbox_xyxy[1] && y <= d.bbox_xyxy[3]
      );
      onHoverBox(hit || null);
    },
    [onHoverBox]
  );

  return (
    <figure className="relative">
      <canvas
        ref={canvasRef}
        onMouseMove={handleMove}
        onMouseLeave={() => onHoverBox?.(null)}
        className="w-full rounded-sm border border-stone-300 bg-stone-100"
      />
      {dims.w > 0 && (
        <figcaption className="mt-2 text-[11px] tabular-nums text-stone-500">
          {dims.w} × {dims.h} px
        </figcaption>
      )}
    </figure>
  );
}

/* ---------------------------------------------------------------------------
 * Main dashboard
 * ------------------------------------------------------------------------- */
export default function InspectionDashboard() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [downloading, setDownloading] = useState(false);

  const [showMask, setShowMask] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.45);

  const [record, setRecord] = useState({ structure_id: "", location: "", inspector: "" });
  const [scaleMode, setScaleMode] = useState("none"); // none | direct | reference
  const [scale, setScale] = useState({ mm_per_pixel: "", reference_length_mm: "", reference_length_px: "" });

  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  const acceptFile = useCallback(
    (f) => {
      if (!f) return;
      if (!f.type.startsWith("image/")) {
        setError("That file is not an image. Upload a JPEG, PNG or TIFF photograph of the surface.");
        return;
      }
      setError(null);
      setResult(null);
      setStatus("idle");
      setFile(f);
      setPreviewUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(f);
      });
    },
    []
  );

  const runInspection = async () => {
    if (!file) return;
    setStatus("running");
    setError(null);

    const form = new FormData();
    form.append("file", file);
    form.append("structure_id", record.structure_id);
    form.append("location", record.location);
    form.append("inspector", record.inspector);
    form.append("return_overlay", "false"); // we composite client-side for the toggles

    if (scaleMode === "direct" && scale.mm_per_pixel) {
      form.append("mm_per_pixel", scale.mm_per_pixel);
    } else if (scaleMode === "reference" && scale.reference_length_mm && scale.reference_length_px) {
      form.append("reference_length_mm", scale.reference_length_mm);
      form.append("reference_length_px", scale.reference_length_px);
    }

    try {
      const res = await fetch(`${API_BASE}/inspect/full`, { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || body.error || `Inspection failed (${res.status})`);
      }
      const data = await res.json();

      // Fetch the mask separately so the canvas can toggle it independently.
      const maskForm = new FormData();
      maskForm.append("file", file);
      const maskRes = await fetch(`${API_BASE}/predict/segment?return_mask=true`, {
        method: "POST",
        body: maskForm,
      });
      const maskData = maskRes.ok ? await maskRes.json() : {};

      setResult({ ...data, mask_png_base64: maskData.mask_png_base64 });
      setStatus("done");
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  };

  const downloadReport = async () => {
    if (!result) return;
    setDownloading(true);
    try {
      const form = new FormData();
      form.append("inspection_id", result.inspection_id);
      form.append("structure_id", record.structure_id);
      form.append("location", record.location);
      form.append("inspector", record.inspector);

      const res = await fetch(`${API_BASE}/reports/pdf`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Report generation failed (${res.status})`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `inspection_${result.inspection_id}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e.message);
    } finally {
      setDownloading(false);
    }
  };

  const assessment = result?.assessment;
  const severity = assessment?.severity;
  const geometry = assessment?.geometry;
  const physical = assessment?.physical;
  const gs = gradeStyle(severity?.grade);

  const detections = result?.detection?.detections || [];
  const classCounts = useMemo(() => {
    const out = {};
    detections.forEach((d) => (out[d.class_name] = (out[d.class_name] || 0) + 1));
    return out;
  }, [detections]);

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900 antialiased">
      {/* ---- Masthead ---------------------------------------------------- */}
      <header className="border-b border-stone-300 bg-white">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-6 py-4">
          <div>
            <h1 className="text-[17px] font-semibold tracking-tight">Concrete damage inspection</h1>
            <p className="mt-0.5 text-[12px] text-stone-500">
              Decision support for structural assessment — findings require engineer verification
            </p>
          </div>
          <div className="flex items-center gap-2 text-[12px]">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                health?.status === "ready"
                  ? "bg-emerald-600"
                  : health?.status === "degraded"
                  ? "bg-amber-500"
                  : "bg-red-600"
              }`}
            />
            <span className="text-stone-600">
              {health?.status === "ready"
                ? "All models loaded"
                : health?.status === "degraded"
                ? "Some models unavailable"
                : "Cannot reach the inspection service"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-6 py-8">
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_400px]">
          {/* ================= Left: the image ============================= */}
          <section>
            {!previewUrl ? (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  acceptFile(e.dataTransfer.files?.[0]);
                }}
                onClick={() => inputRef.current?.click()}
                className={`flex min-h-[460px] cursor-pointer flex-col items-center justify-center rounded-sm border-2 border-dashed px-8 text-center transition-colors ${
                  dragging ? "border-stone-900 bg-stone-100" : "border-stone-300 bg-white hover:border-stone-500"
                }`}
              >
                <p className="text-[15px] font-medium text-stone-800">
                  Drop a photograph of the concrete surface here
                </p>
                <p className="mt-2 max-w-sm text-[13px] leading-relaxed text-stone-500">
                  Photograph the defect square-on and include a ruler or target of known size in the
                  same plane — without one, the system reports relative severity only.
                </p>
                <span className="mt-5 rounded-sm bg-stone-900 px-4 py-2 text-[13px] font-medium text-white">
                  Choose an image
                </span>
              </div>
            ) : (
              <>
                <AnnotatedCanvas
                  imageUrl={previewUrl}
                  maskBase64={result?.mask_png_base64}
                  detections={detections}
                  showMask={showMask && !!result}
                  showBoxes={showBoxes && !!result}
                  maskOpacity={maskOpacity}
                  onHoverBox={setHovered}
                />

                <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3">
                  <Toggle checked={showMask} onChange={setShowMask}>
                    Damage surface
                  </Toggle>
                  <Toggle checked={showBoxes} onChange={setShowBoxes}>
                    Defect regions
                  </Toggle>
                  <label className="flex items-center gap-2 text-[13px] text-stone-700">
                    Overlay strength
                    <input
                      type="range"
                      min="0.15"
                      max="0.85"
                      step="0.05"
                      value={maskOpacity}
                      onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                      className="w-28 accent-stone-900"
                    />
                  </label>
                  <button
                    onClick={() => {
                      setFile(null);
                      setPreviewUrl(null);
                      setResult(null);
                      setStatus("idle");
                    }}
                    className="ml-auto text-[13px] text-stone-600 underline underline-offset-2 hover:text-stone-900"
                  >
                    Use a different image
                  </button>
                </div>

                {hovered && (
                  <p className="mt-3 rounded-sm bg-stone-900 px-3 py-2 text-[12px] tabular-nums text-stone-100">
                    {hovered.class_name} · {(hovered.confidence * 100).toFixed(1)}% confidence ·{" "}
                    {hovered.area_px.toLocaleString()} px² · {(hovered.area_ratio * 100).toFixed(2)}% of frame
                  </p>
                )}
              </>
            )}

            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => acceptFile(e.target.files?.[0])}
            />
          </section>

          {/* ================= Right: the record ========================== */}
          <aside className="lg:sticky lg:top-8 lg:self-start">
            {status === "done" && severity ? (
              <>
                <div className={`${gs.bg} rounded-sm px-5 py-4 text-white`}>
                  <p className="text-[19px] font-semibold leading-tight">{severity.grade}</p>
                  <p className="mt-1 text-[12px] tabular-nums text-white/85">
                    Severity index {severity.rdsi.toFixed(1)} / 100 · graded on {severity.grading_basis} criteria
                  </p>
                </div>

                <SectionHeading note={physical?.scale_available ? "with scale" : "pixels only"}>
                  Measurements
                </SectionHeading>
                <div>
                  <Figure
                    label="Surface affected"
                    value={geometry.surface_coverage_percent.toFixed(3)}
                    unit="%"
                  />
                  <Figure label="Damaged area" value={geometry.damaged_area_px.toLocaleString()} unit="px²" />
                  <Figure label="Separate defects" value={geometry.num_regions} />
                  <Figure
                    label="Longest defect span"
                    value={geometry.max_defect_span_px.toFixed(0)}
                    unit="px"
                  />
                  <Figure label="Maximum width" value={geometry.max_width_px.toFixed(2)} unit="px" />
                  {physical?.scale_available ? (
                    <>
                      <Figure
                        label="Maximum crack width"
                        value={physical.max_crack_width_mm.toFixed(2)}
                        unit="mm"
                      />
                      <Figure
                        label="Damaged area"
                        value={physical.damaged_area_cm2.toLocaleString(undefined, {
                          maximumFractionDigits: 2,
                        })}
                        unit="cm²"
                      />
                      <Figure
                        label="Longest span"
                        value={physical.max_defect_span_mm.toLocaleString(undefined, {
                          maximumFractionDigits: 0,
                        })}
                        unit="mm"
                      />
                    </>
                  ) : (
                    <p className="mt-3 border-l-2 border-stone-300 pl-3 text-[12px] leading-relaxed text-stone-600">
                      No scale reference was supplied, so dimensions stay in pixels and cannot be
                      compared between photographs taken at different distances.
                    </p>
                  )}
                </div>

                <SectionHeading note={`${result.detection.num_detections} found`}>
                  Detected defects
                </SectionHeading>
                {Object.keys(classCounts).length ? (
                  <div>
                    {Object.entries(classCounts).map(([name, n]) => (
                      <Figure key={name} label={name} value={n} />
                    ))}
                    <Figure
                      label="Mean confidence"
                      value={(assessment.detection_summary.mean_confidence * 100).toFixed(1)}
                      unit="%"
                    />
                  </div>
                ) : (
                  <p className="text-[13px] text-stone-600">
                    The detector localised no defect regions in this frame.
                  </p>
                )}

                <SectionHeading>How this grade was reached</SectionHeading>
                <ul className="space-y-2">
                  {severity.reasoning.map((r, i) => (
                    <li key={i} className="text-[12.5px] leading-relaxed text-stone-700">
                      {r}
                    </li>
                  ))}
                </ul>
                {severity.confidence_notes?.length > 0 && (
                  <div className="mt-3 border-l-2 border-amber-500 pl-3">
                    {severity.confidence_notes.map((n, i) => (
                      <p key={i} className="text-[12px] leading-relaxed text-stone-600">
                        {n}
                      </p>
                    ))}
                  </div>
                )}

                <SectionHeading note={assessment.recommendations.priority}>
                  Recommended action
                </SectionHeading>
                <p className="mb-3 text-[12.5px] text-stone-600">
                  Re-inspect within {assessment.recommendations.inspection_interval.toLowerCase()}.
                </p>
                <ol className="space-y-2">
                  {assessment.recommendations.actions.map((a, i) => (
                    <li key={i} className="flex gap-3 text-[12.5px] leading-relaxed text-stone-700">
                      <span className="tabular-nums text-stone-400">{i + 1}</span>
                      <span>{a}</span>
                    </li>
                  ))}
                </ol>

                <button
                  onClick={downloadReport}
                  disabled={downloading}
                  className="mt-7 w-full rounded-sm bg-stone-900 px-4 py-3 text-[14px] font-medium text-white transition-colors hover:bg-stone-700 disabled:cursor-not-allowed disabled:bg-stone-400"
                >
                  {downloading ? "Preparing report…" : "Download inspection report"}
                </button>
                <p className="mt-3 text-[11px] leading-relaxed text-stone-500">{result.disclaimer}</p>
              </>
            ) : (
              <>
                <SectionHeading>Inspection record</SectionHeading>
                <div className="space-y-3">
                  {[
                    ["structure_id", "Structure reference", "e.g. Bridge B-14, Pier 3"],
                    ["location", "Location", "e.g. North abutment, west face"],
                    ["inspector", "Inspector", "Your name"],
                  ].map(([key, label, placeholder]) => (
                    <div key={key}>
                      <label className="block text-[12px] text-stone-600">{label}</label>
                      <input
                        value={record[key]}
                        onChange={(e) => setRecord({ ...record, [key]: e.target.value })}
                        placeholder={placeholder}
                        className="mt-1 w-full rounded-sm border border-stone-300 bg-white px-3 py-2 text-[13px] placeholder:text-stone-400 focus:border-stone-900 focus:outline-none focus:ring-1 focus:ring-stone-900"
                      />
                    </div>
                  ))}
                </div>

                <SectionHeading note="optional">Scale reference</SectionHeading>
                <p className="mb-3 text-[12.5px] leading-relaxed text-stone-600">
                  Supply a scale to get millimetre measurements and physical grading. Without one the
                  grade is a relative index for triage.
                </p>
                <div className="space-y-2">
                  {[
                    ["none", "No scale — relative severity only"],
                    ["direct", "I know the millimetres per pixel"],
                    ["reference", "There is an object of known size in the photo"],
                  ].map(([val, label]) => (
                    <label key={val} className="flex cursor-pointer items-start gap-2 text-[13px] text-stone-700">
                      <input
                        type="radio"
                        name="scaleMode"
                        checked={scaleMode === val}
                        onChange={() => setScaleMode(val)}
                        className="mt-0.5 text-stone-900 focus:ring-stone-900"
                      />
                      {label}
                    </label>
                  ))}
                </div>

                {scaleMode === "direct" && (
                  <input
                    type="number"
                    step="0.0001"
                    value={scale.mm_per_pixel}
                    onChange={(e) => setScale({ ...scale, mm_per_pixel: e.target.value })}
                    placeholder="Millimetres per pixel"
                    className="mt-3 w-full rounded-sm border border-stone-300 px-3 py-2 text-[13px] tabular-nums focus:border-stone-900 focus:outline-none focus:ring-1 focus:ring-stone-900"
                  />
                )}
                {scaleMode === "reference" && (
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      value={scale.reference_length_mm}
                      onChange={(e) => setScale({ ...scale, reference_length_mm: e.target.value })}
                      placeholder="Real length (mm)"
                      className="rounded-sm border border-stone-300 px-3 py-2 text-[13px] tabular-nums focus:border-stone-900 focus:outline-none focus:ring-1 focus:ring-stone-900"
                    />
                    <input
                      type="number"
                      value={scale.reference_length_px}
                      onChange={(e) => setScale({ ...scale, reference_length_px: e.target.value })}
                      placeholder="Measured (px)"
                      className="rounded-sm border border-stone-300 px-3 py-2 text-[13px] tabular-nums focus:border-stone-900 focus:outline-none focus:ring-1 focus:ring-stone-900"
                    />
                  </div>
                )}

                <button
                  onClick={runInspection}
                  disabled={!file || status === "running"}
                  className="mt-7 w-full rounded-sm bg-stone-900 px-4 py-3 text-[14px] font-medium text-white transition-colors hover:bg-stone-700 disabled:cursor-not-allowed disabled:bg-stone-300"
                >
                  {status === "running"
                    ? "Inspecting…"
                    : file
                    ? "Run inspection"
                    : "Add an image to begin"}
                </button>
              </>
            )}

            {error && (
              <div className="mt-4 rounded-sm border border-red-300 bg-red-50 px-4 py-3">
                <p className="text-[13px] font-medium text-red-900">Inspection did not complete</p>
                <p className="mt-1 text-[12.5px] leading-relaxed text-red-800">{error}</p>
              </div>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}