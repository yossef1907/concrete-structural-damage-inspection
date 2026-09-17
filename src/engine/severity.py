"""
=============================================================================
08_SEVERITY_ASSESSMENT.py   →   deploy as  src/engine/severity.py
Concrete Structural Damage Inspection System — Phase 3, Stage 2
=============================================================================
Heuristic severity engine. Consumes the detection and segmentation contracts
from notebooks 05 and 06 and produces a structured, JSON-serialisable
assessment for the API and the PDF report.

Design position, stated plainly because it belongs in the report:
  Pixel geometry is not a structural quantity. A 2 mm crack photographed at
  0.5 m and a 20 mm crack photographed at 5 m can occupy identical pixels.
  Therefore:
    * Without a scale reference, this engine emits RDSI — a Relative Damage
      Severity Index in [0, 100] — and labels every physical field "unavailable".
    * With a pixel-to-millimetre scale, it additionally emits mm² area and mm
      crack width, and grading switches to the physical thresholds, which are
      the ones an engineer can defend.
  The engine never silently mixes the two. `scale_available` travels with every
  result and belongs on the face of the report.

This file is import-safe: the demo cells at the bottom run only under
`if __name__ == "__main__"`, so `from engine.severity import SeverityEngine`
executes no I/O.
=============================================================================
"""

# %%
# =============================================================================
# CELL 1 — IMPORTS & GRADING CONFIGURATION
# =============================================================================
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence, Any

import numpy as np
import cv2


class SeverityGrade(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"
    NONE = "No damage detected"


# Ordered worst-first for max() style resolution.
GRADE_ORDER = [SeverityGrade.NONE, SeverityGrade.LOW, SeverityGrade.MEDIUM,
               SeverityGrade.HIGH, SeverityGrade.CRITICAL]


@dataclass
class SeverityThresholds:
    """Grade boundaries.

    RDSI bands are relative and tuned for triage — they decide which images an
    engineer opens first, nothing more.

    The crack-width bands follow the widely used serviceability convention for
    reinforced concrete in non-aggressive exposure (roughly ACI 224R territory):
    below ~0.3 mm is generally a serviceability observation; 0.3-1.0 mm warrants
    monitoring and sealing; above ~1.0 mm suggests structural investigation.
    They are defaults, not code compliance — set them from your project's own
    specification before any report leaves the building.
    """
    # --- Relative (no scale) — RDSI bands ----------------------------------
    # Calibrated against the synthetic ladder in CELL 5: a hairline crack
    # crossing the frame must grade Low, and a wide crack covering >6% of the
    # surface must grade High. Re-run that ladder after changing these.
    rdsi_low: float = 12.0
    rdsi_medium: float = 30.0
    rdsi_high: float = 55.0

    # --- Physical (scale supplied) — max crack width, mm -------------------
    width_mm_low: float = 0.30
    width_mm_medium: float = 1.00
    width_mm_high: float = 3.00

    # --- Physical — damaged surface area, percentage of frame --------------
    area_pct_low: float = 1.0
    area_pct_medium: float = 5.0
    area_pct_high: float = 15.0

    # --- Contributing factors ----------------------------------------------
    density_high: float = 0.35      # bbox coverage ratio that signals spread
    region_count_high: int = 12     # many separate defects = distributed damage
    spalling_class_names: Sequence[str] = ("spalling", "spall", "delamination")


@dataclass
class RDSIWeights:
    """RDSI = weighted blend of four normalised signals, each in [0,1].

    Coverage dominates because it is the most stable signal across image scales.
    Span and density capture 'one big defect' vs 'many small ones', which pure
    coverage cannot distinguish. Confidence damps the score when the models
    themselves are unsure, so a low-confidence frame does not escalate on its own.
    """
    coverage: float = 0.55
    span: float = 0.20
    density: float = 0.20
    confidence: float = 0.05

    # Saturation points — the value at which each signal counts as "1.0".
    # span_saturation is near a full diagonal deliberately: most real cracks
    # traverse a large part of the frame, so a lower value would saturate the
    # span term on almost every image and drown out coverage.
    coverage_saturation: float = 0.15     # 15% of the frame damaged is severe
    span_saturation: float = 0.95         # longest defect spans 95% of the diagonal
    density_saturation: float = 0.50      # bboxes cover 50% of the frame


# %%
# =============================================================================
# CELL 2 — GEOMETRY HELPERS
# =============================================================================
def _largest_span_px(mask_bin: np.ndarray) -> tuple[float, dict]:
    """Longest straight-line extent of any single connected defect, in pixels.

    Uses the minimum-area rectangle per component rather than the bounding box,
    so a diagonal crack is measured along its own axis instead of being inflated
    by an axis-aligned box.
    """
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_span, best_info = 0.0, {}
    for c in contours:
        if len(c) < 2:
            continue
        (cx, cy), (w, h), angle = cv2.minAreaRect(c)
        span = float(max(w, h))
        if span > best_span:
            best_span = span
            best_info = {
                "centroid_px": [round(float(cx), 2), round(float(cy), 2)],
                "length_px": round(span, 2),
                "thickness_px": round(float(min(w, h)), 2),
                "orientation_deg": round(float(angle), 2),
                "contour_area_px": round(float(cv2.contourArea(c)), 2),
            }
    return best_span, best_info


def _width_stats_px(mask_bin: np.ndarray) -> dict:
    """Crack width from the distance transform.

    For a thin elongated region, the distance transform peaks at the medial
    axis and its value there is half the local width. Doubling the peak gives
    maximum width; doubling the mean over skeleton-adjacent pixels gives a
    representative width. This is far more faithful for cracks than
    area/length, which collapses on branched or curved defects.
    """
    if mask_bin.max() == 0:
        return {"max_width_px": 0.0, "mean_width_px": 0.0, "p95_width_px": 0.0}
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    ridge = dist[dist > 0]
    if ridge.size == 0:
        return {"max_width_px": 0.0, "mean_width_px": 0.0, "p95_width_px": 0.0}
    # Only the upper part of the distance distribution lies near the medial axis;
    # the lower part is edge pixels and would bias the mean toward zero.
    cutoff = np.percentile(ridge, 70)
    core = ridge[ridge >= cutoff]
    return {
        "max_width_px": round(float(ridge.max() * 2.0), 3),
        "mean_width_px": round(float(core.mean() * 2.0), 3),
        "p95_width_px": round(float(np.percentile(ridge, 95) * 2.0), 3),
    }


def _bbox_density(boxes: Sequence[dict], width: int, height: int) -> dict:
    """Union area of all boxes over frame area — union, not sum, so overlapping
    detections of the same defect do not double-count."""
    frame = float(width * height)
    if not boxes or frame <= 0:
        return {"bbox_coverage_ratio": 0.0, "bbox_union_px": 0.0,
                "bbox_sum_px": 0.0, "bbox_overlap_factor": 0.0}

    canvas = np.zeros((height, width), dtype=np.uint8)
    total = 0.0
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b["bbox_xyxy"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = 1
            total += (x2 - x1) * (y2 - y1)

    union = float(canvas.sum())
    return {
        "bbox_coverage_ratio": round(union / frame, 6),
        "bbox_union_px": round(union, 1),
        "bbox_sum_px": round(total, 1),
        "bbox_overlap_factor": round(total / union, 3) if union > 0 else 0.0,
    }


def _region_profile(mask_bin: np.ndarray, top_k: int = 10) -> dict:
    """Per-defect inventory — the table that goes into the PDF appendix."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, 8)
    regions = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        regions.append({
            "region_id": i,
            "area_px": area,
            "bbox_xywh": [int(stats[i, cv2.CC_STAT_LEFT]),
                          int(stats[i, cv2.CC_STAT_TOP]), w, h],
            "centroid_px": [round(float(centroids[i][0]), 1),
                            round(float(centroids[i][1]), 1)],
            "elongation": round(max(w, h) / max(min(w, h), 1), 2),
        })
    regions.sort(key=lambda r: -r["area_px"])
    return {
        "num_regions": len(regions),
        "largest_regions": regions[:top_k],
        "area_px_total": int(sum(r["area_px"] for r in regions)),
    }


def _norm(value: float, saturation: float) -> float:
    if saturation <= 0:
        return 0.0
    return float(np.clip(value / saturation, 0.0, 1.0))


# %%
# =============================================================================
# CELL 3 — THE ENGINE
# =============================================================================
@dataclass
class ScaleReference:
    """Pixel-to-millimetre conversion.

    Supply exactly one of:
      * mm_per_pixel — measured directly (e.g. a ruler or target in frame)
      * reference_length_mm + reference_length_px — a known object measured
        on the image
      * camera_distance_mm + focal_length_px — pinhole estimate; least reliable,
        and flagged as such in the output
    """
    mm_per_pixel: Optional[float] = None
    reference_length_mm: Optional[float] = None
    reference_length_px: Optional[float] = None
    camera_distance_mm: Optional[float] = None
    focal_length_px: Optional[float] = None
    note: str = ""

    def resolve(self) -> tuple[Optional[float], str]:
        """Returns (mm_per_pixel, method) or (None, reason)."""
        if self.mm_per_pixel and self.mm_per_pixel > 0:
            return float(self.mm_per_pixel), "direct"
        if (self.reference_length_mm and self.reference_length_px
                and self.reference_length_px > 0):
            return (float(self.reference_length_mm) / float(self.reference_length_px),
                    "reference_object")
        if (self.camera_distance_mm and self.focal_length_px
                and self.focal_length_px > 0):
            return (float(self.camera_distance_mm) / float(self.focal_length_px),
                    "pinhole_estimate")
        return None, "no_scale_supplied"


class SeverityEngine:
    """Turns model outputs into a defensible, structured severity assessment."""

    VERSION = "1.0.0"

    def __init__(self,
                 thresholds: Optional[SeverityThresholds] = None,
                 weights: Optional[RDSIWeights] = None):
        self.th = thresholds or SeverityThresholds()
        self.w = weights or RDSIWeights()

    # -- RDSI ---------------------------------------------------------------
    def _rdsi(self, coverage_ratio: float, span_ratio: float,
              bbox_density: float, mean_conf: float) -> dict:
        c = _norm(coverage_ratio, self.w.coverage_saturation)
        s = _norm(span_ratio, self.w.span_saturation)
        d = _norm(bbox_density, self.w.density_saturation)
        conf = float(np.clip(mean_conf, 0.0, 1.0))

        raw = (self.w.coverage * c + self.w.span * s +
               self.w.density * d + self.w.confidence * conf)
        total_w = self.w.coverage + self.w.span + self.w.density + self.w.confidence
        rdsi = 100.0 * raw / max(total_w, 1e-9)

        return {
            "rdsi": round(float(np.clip(rdsi, 0.0, 100.0)), 2),
            "components": {
                "coverage_normalised": round(c, 4),
                "span_normalised": round(s, 4),
                "density_normalised": round(d, 4),
                "confidence": round(conf, 4),
            },
            "weights": asdict(self.w),
        }

    # -- Grading ------------------------------------------------------------
    def _grade_relative(self, rdsi: float, coverage_ratio: float) -> tuple[SeverityGrade, list[str]]:
        reasons = []
        if coverage_ratio <= 0 and rdsi <= 0:
            return SeverityGrade.NONE, ["No damage pixels segmented."]
        if rdsi >= self.th.rdsi_high:
            g = SeverityGrade.CRITICAL
        elif rdsi >= self.th.rdsi_medium:
            g = SeverityGrade.HIGH
        elif rdsi >= self.th.rdsi_low:
            g = SeverityGrade.MEDIUM
        else:
            g = SeverityGrade.LOW
        reasons.append(f"RDSI {rdsi:.1f} falls in the {g.value} band "
                       f"(bands: <{self.th.rdsi_low} Low, <{self.th.rdsi_medium} Medium, "
                       f"<{self.th.rdsi_high} High, ≥{self.th.rdsi_high} Critical).")
        reasons.append("Grade is relative — no scale reference was supplied, so it "
                       "reflects how much of this frame is affected, not physical "
                       "crack size.")
        return g, reasons

    def _grade_physical(self, max_width_mm: float, area_pct: float) -> tuple[SeverityGrade, list[str]]:
        reasons = []
        if max_width_mm >= self.th.width_mm_high:
            gw = SeverityGrade.CRITICAL
        elif max_width_mm >= self.th.width_mm_medium:
            gw = SeverityGrade.HIGH
        elif max_width_mm >= self.th.width_mm_low:
            gw = SeverityGrade.MEDIUM
        else:
            gw = SeverityGrade.LOW
        reasons.append(f"Maximum crack width {max_width_mm:.2f} mm → {gw.value} "
                       f"(bands: <{self.th.width_mm_low} Low, "
                       f"<{self.th.width_mm_medium} Medium, "
                       f"<{self.th.width_mm_high} High).")

        if area_pct >= self.th.area_pct_high:
            ga = SeverityGrade.CRITICAL
        elif area_pct >= self.th.area_pct_medium:
            ga = SeverityGrade.HIGH
        elif area_pct >= self.th.area_pct_low:
            ga = SeverityGrade.MEDIUM
        else:
            ga = SeverityGrade.LOW
        reasons.append(f"Damaged surface {area_pct:.2f}% of frame → {ga.value}.")

        # Governing grade is the worse of the two — width and extent are both
        # independently sufficient to escalate.
        g = max(gw, ga, key=lambda x: GRADE_ORDER.index(x))
        reasons.append(f"Governing grade is the more severe of the two: {g.value}.")
        return g, reasons

    def _escalate(self, grade: SeverityGrade, detections: Sequence[dict],
                  density: float, num_regions: int) -> tuple[SeverityGrade, list[str]]:
        """Contributing factors that raise the grade by one step."""
        notes = []
        idx = GRADE_ORDER.index(grade)

        names = {str(d.get("class_name", "")).lower() for d in detections}
        if names & set(self.th.spalling_class_names):
            if idx < len(GRADE_ORDER) - 1:
                idx += 1
                notes.append("Spalling or delamination detected — section loss "
                             "exposes reinforcement, so the grade is raised one step.")
        if density >= self.th.density_high:
            if idx < len(GRADE_ORDER) - 1:
                idx += 1
                notes.append(f"Damage is spread across {density:.0%} of the frame "
                             f"rather than localised — grade raised one step.")
        if num_regions >= self.th.region_count_high:
            if idx < len(GRADE_ORDER) - 1:
                idx += 1
                notes.append(f"{num_regions} separate defect regions indicate "
                             f"distributed deterioration — grade raised one step.")
        return GRADE_ORDER[idx], notes

    # -- Recommendations ----------------------------------------------------
    def _recommendations(self, grade: SeverityGrade, scale_available: bool,
                         detections: Sequence[dict]) -> dict:
        has_spalling = bool({str(d.get("class_name", "")).lower()
                             for d in detections} & set(self.th.spalling_class_names))

        table = {
            SeverityGrade.NONE: {
                "priority": "Routine",
                "inspection_interval": "Next scheduled cycle",
                "actions": [
                    "No defect surface identified in this frame.",
                    "Retain the image in the baseline record for future comparison.",
                ],
            },
            SeverityGrade.LOW: {
                "priority": "Routine",
                "inspection_interval": "12 months",
                "actions": [
                    "Record location and photograph with a scale reference in frame.",
                    "Re-image at the next cycle and compare extent against this baseline.",
                    "No intervention required on this evidence alone.",
                ],
            },
            SeverityGrade.MEDIUM: {
                "priority": "Planned maintenance",
                "inspection_interval": "6 months",
                "actions": [
                    "Install crack gauges or datum marks to track movement.",
                    "Seal surface cracks to limit water and chloride ingress.",
                    "Check drainage and exposure conditions near the defect.",
                    "Have a qualified engineer confirm the defect is non-structural.",
                ],
            },
            SeverityGrade.HIGH: {
                "priority": "Prompt engineering review",
                "inspection_interval": "1-3 months",
                "actions": [
                    "Arrange a site inspection by a certified structural engineer.",
                    "Carry out cover meter and half-cell survey to assess reinforcement.",
                    "Monitor crack width with fixed gauges; log readings monthly.",
                    "Plan repair: crack injection, patch repair, or section restoration.",
                    "Review loading on the affected member pending assessment.",
                ],
            },
            SeverityGrade.CRITICAL: {
                "priority": "Immediate action",
                "inspection_interval": "Immediate, then continuous monitoring",
                "actions": [
                    "Escalate to a certified structural engineer without delay.",
                    "Restrict access to the affected zone until the member is assessed.",
                    "Consider temporary propping or load reduction on engineer's advice.",
                    "Carry out structural capacity assessment before any repair design.",
                    "Document with a full photographic and dimensional survey.",
                ],
            },
        }
        rec = dict(table[grade])
        rec["actions"] = list(rec["actions"])

        if has_spalling:
            rec["actions"].append(
                "Spalling present: check for exposed or corroded reinforcement and "
                "assess remaining section before specifying repair.")
        if not scale_available:
            rec["actions"].append(
                "No scale reference was supplied. Re-photograph with a ruler or "
                "target in frame before using any dimension from this report.")
        return rec

    # -- Public API ---------------------------------------------------------
    def assess(self,
               mask: np.ndarray,
               detections: Optional[Sequence[dict]] = None,
               image_shape: Optional[tuple[int, int]] = None,
               scale: Optional[ScaleReference] = None,
               image_id: str = "") -> dict:
        """Compute the full severity assessment.

        mask        : uint8 {0,255} or bool array at ORIGINAL image resolution
                      (exactly what predict_mask() returns under key "mask").
        detections  : the "detections" list from predict_boxes(); may be empty.
        image_shape : (height, width); inferred from mask when omitted.
        scale       : optional ScaleReference for physical units.
        """
        detections = list(detections or [])

        m = np.asarray(mask)
        if m.dtype != np.uint8:
            m = (m > 0).astype(np.uint8) * 255
        m = ((m > 127).astype(np.uint8)) * 255

        H, W = image_shape if image_shape else m.shape[:2]
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        frame_px = int(H * W)
        diagonal_px = math.hypot(H, W)

        # --- core pixel metrics --------------------------------------------
        damaged_px = int((m > 0).sum())
        coverage_ratio = damaged_px / max(frame_px, 1)
        span_px, span_info = _largest_span_px(m)
        widths = _width_stats_px(m)
        regions = _region_profile(m)
        density = _bbox_density(detections, W, H)

        mean_conf = (float(np.mean([d.get("confidence", 0.0) for d in detections]))
                     if detections else 0.0)
        max_conf = (float(np.max([d.get("confidence", 0.0) for d in detections]))
                    if detections else 0.0)

        # --- scale resolution ------------------------------------------------
        mm_per_px, scale_method = (scale.resolve() if scale else (None, "no_scale_supplied"))
        scale_available = mm_per_px is not None

        physical: dict[str, Any] = {
            "scale_available": scale_available,
            "scale_method": scale_method,
            "mm_per_pixel": round(mm_per_px, 6) if scale_available else None,
        }
        if scale_available:
            mm2_per_px = mm_per_px ** 2
            physical.update({
                "damaged_area_mm2": round(damaged_px * mm2_per_px, 2),
                "damaged_area_cm2": round(damaged_px * mm2_per_px / 100.0, 3),
                "max_crack_width_mm": round(widths["max_width_px"] * mm_per_px, 3),
                "mean_crack_width_mm": round(widths["mean_width_px"] * mm_per_px, 3),
                "p95_crack_width_mm": round(widths["p95_width_px"] * mm_per_px, 3),
                "max_defect_span_mm": round(span_px * mm_per_px, 2),
                "frame_area_mm2": round(frame_px * mm2_per_px, 1),
            })
            if scale_method == "pinhole_estimate":
                physical["caveat"] = (
                    "Scale derived from camera distance and focal length. This "
                    "assumes the surface is planar and perpendicular to the lens; "
                    "treat dimensions as approximate.")
        else:
            physical.update({
                "damaged_area_mm2": None, "damaged_area_cm2": None,
                "max_crack_width_mm": None, "mean_crack_width_mm": None,
                "p95_crack_width_mm": None, "max_defect_span_mm": None,
                "frame_area_mm2": None,
                "caveat": ("No scale reference supplied. All dimensions are in "
                           "pixels and are not comparable between images."),
            })

        # --- RDSI and grading -------------------------------------------------
        span_ratio = span_px / max(diagonal_px, 1e-9)
        rdsi_block = self._rdsi(coverage_ratio, span_ratio,
                                density["bbox_coverage_ratio"], mean_conf)
        rdsi = rdsi_block["rdsi"]

        if damaged_px == 0 and not detections:
            base_grade = SeverityGrade.NONE
            reasons = ["No damage pixels segmented and no defects detected."]
            grading_basis = "none"
        elif scale_available:
            base_grade, reasons = self._grade_physical(
                physical["max_crack_width_mm"], coverage_ratio * 100.0)
            grading_basis = "physical"
        else:
            base_grade, reasons = self._grade_relative(rdsi, coverage_ratio)
            grading_basis = "relative"

        grade, escalation_notes = self._escalate(
            base_grade, detections, density["bbox_coverage_ratio"],
            regions["num_regions"])

        # --- detection summary -------------------------------------------------
        by_class: dict[str, int] = {}
        for d in detections:
            k = str(d.get("class_name", "unknown"))
            by_class[k] = by_class.get(k, 0) + 1

        # --- confidence in the assessment itself -------------------------------
        # Low model confidence or a tiny damaged area means the geometry is
        # poorly determined; the report should say so rather than imply precision.
        assessment_confidence = "high"
        confidence_notes = []
        if damaged_px and damaged_px < 200:
            assessment_confidence = "low"
            confidence_notes.append(
                "Damaged area is under 200 px; width and span estimates are "
                "close to the resolution limit.")
        if detections and mean_conf < 0.4:
            assessment_confidence = "low" if assessment_confidence == "low" else "moderate"
            confidence_notes.append(
                f"Mean detection confidence is {mean_conf:.2f}; localisation may "
                f"be unreliable.")
        if not detections and damaged_px > 0:
            assessment_confidence = "moderate" if assessment_confidence == "high" else assessment_confidence
            confidence_notes.append(
                "Segmentation found damage but the detector localised none; the "
                "two models disagree on this frame.")

        return {
            "engine_version": self.VERSION,
            "image_id": image_id,
            "image_size": {"width": int(W), "height": int(H),
                           "total_pixels": frame_px,
                           "diagonal_px": round(diagonal_px, 1)},

            "geometry": {
                "damaged_area_px": damaged_px,
                "surface_coverage_ratio": round(coverage_ratio, 6),
                "surface_coverage_percent": round(coverage_ratio * 100.0, 4),
                "max_defect_span_px": round(span_px, 2),
                "max_defect_span_ratio": round(span_ratio, 4),
                "largest_defect": span_info,
                **widths,
                "num_regions": regions["num_regions"],
                "largest_regions": regions["largest_regions"],
            },

            "detection_summary": {
                "num_detections": len(detections),
                "by_class": by_class,
                "mean_confidence": round(mean_conf, 4),
                "max_confidence": round(max_conf, 4),
                **density,
            },

            "physical": physical,

            "severity": {
                "grade": grade.value,
                "grading_basis": grading_basis,
                "rdsi": rdsi,
                "rdsi_breakdown": rdsi_block,
                "base_grade_before_escalation": base_grade.value,
                "reasoning": reasons + escalation_notes,
                "assessment_confidence": assessment_confidence,
                "confidence_notes": confidence_notes,
                "thresholds_used": asdict(self.th),
            },

            "recommendations": self._recommendations(grade, scale_available, detections),

            "disclaimer": (
                "This assessment is generated by an automated decision-support "
                "system and is not a structural engineering determination. All "
                "findings require verification by a certified structural engineer "
                "before any maintenance, repair, or access decision is made."),
        }


# %%
# =============================================================================
# CELL 4 — VISUAL OVERLAY FOR THE REPORT
# =============================================================================
GRADE_COLOURS = {
    SeverityGrade.NONE.value: (110, 190, 120),
    SeverityGrade.LOW.value: (95, 175, 235),
    SeverityGrade.MEDIUM.value: (250, 190, 60),
    SeverityGrade.HIGH.value: (245, 130, 45),
    SeverityGrade.CRITICAL.value: (225, 55, 60),
}


def render_overlay(image_rgb: np.ndarray,
                   mask: Optional[np.ndarray] = None,
                   detections: Optional[Sequence[dict]] = None,
                   assessment: Optional[dict] = None,
                   mask_alpha: float = 0.45) -> np.ndarray:
    """Composite mask + boxes + a grade banner onto the image. Returns RGB uint8.

    The banner is the only coloured chrome — the grade is the one thing an
    engineer scanning a hundred report pages needs to read at a glance.
    """
    out = np.ascontiguousarray(image_rgb.copy())
    H, W = out.shape[:2]

    if mask is not None:
        m = np.asarray(mask)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        sel = m > 127 if m.dtype == np.uint8 else m > 0
        if sel.any():
            tint = np.array([230, 60, 60], dtype=np.float32)
            out[sel] = ((1 - mask_alpha) * out[sel] + mask_alpha * tint).astype(np.uint8)
            edges = cv2.Canny((sel.astype(np.uint8)) * 255, 50, 150)
            out[edges > 0] = (255, 235, 235)

    for d in (detections or []):
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox_xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 210, 60), 2)
        label = f"{d.get('class_name','damage')} {d.get('confidence',0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (255, 210, 60), -1)
        cv2.putText(out, label, (x1 + 3, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)

    if assessment is not None:
        grade = assessment["severity"]["grade"]
        colour = GRADE_COLOURS.get(grade, (120, 120, 120))
        bar_h = max(34, H // 18)
        band = out[:bar_h].astype(np.float32)
        out[:bar_h] = (0.25 * band + 0.75 * np.array(colour, np.float32)).astype(np.uint8)
        cov = assessment["geometry"]["surface_coverage_percent"]
        rdsi = assessment["severity"]["rdsi"]
        text = f"{grade}  ·  RDSI {rdsi:.1f}  ·  {cov:.2f}% of surface"
        cv2.putText(out, text, (12, int(bar_h * 0.68)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, bar_h / 60), (20, 20, 20),
                    2, cv2.LINE_AA)
    return out


def assessment_to_json(assessment: dict, indent: int = 2) -> str:
    """Safe JSON dump — numpy scalars and arrays are converted, not crashed on."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not JSON serialisable: {type(o)}")
    return json.dumps(assessment, indent=indent, default=_default)


# %%
# =============================================================================
# CELL 5 — DEMO / VALIDATION  (runs only when executed directly)
# =============================================================================
if __name__ == "__main__":
    import sys

    engine = SeverityEngine()

    # --- Synthetic cases: verify the grade ladder behaves monotonically -----
    def synth(width_px: int, length_frac: float, size: int = 800) -> np.ndarray:
        m = np.zeros((size, size), np.uint8)
        cv2.line(m, (int(size * 0.1), int(size * 0.2)),
                 (int(size * 0.1 + size * length_frac), int(size * 0.8)),
                 255, width_px)
        return m

    print("=" * 78)
    print("SYNTHETIC LADDER — relative grading, no scale")
    print("=" * 78)
    for w, lf, label in ((2, 0.2, "hairline, short"),
                         (6, 0.5, "moderate"),
                         (18, 0.8, "wide, long"),
                         (45, 0.95, "severe")):
        a = engine.assess(synth(w, lf), detections=[], image_id=label)
        print(f"{label:18s} coverage={a['geometry']['surface_coverage_percent']:6.3f}%  "
              f"span={a['geometry']['max_defect_span_px']:7.1f}px  "
              f"RDSI={a['severity']['rdsi']:6.2f}  → {a['severity']['grade']}")

    print("\n" + "=" * 78)
    print("SAME DEFECT, WITH SCALE — grading switches to physical thresholds")
    print("=" * 78)
    mask = synth(8, 0.6)
    for mmpp, note in ((0.05, "close-up, 0.05 mm/px"),
                       (0.25, "mid range, 0.25 mm/px"),
                       (1.20, "far standoff, 1.2 mm/px")):
        a = engine.assess(mask, detections=[],
                          scale=ScaleReference(mm_per_pixel=mmpp), image_id=note)
        print(f"{note:26s} max width = {a['physical']['max_crack_width_mm']:6.2f} mm  "
              f"→ {a['severity']['grade']}  (basis: {a['severity']['grading_basis']})")
    print("\nIdentical pixels, three different verdicts. That is exactly why the "
          "engine refuses to report millimetres without a scale reference.")

    print("\n" + "=" * 78)
    print("ESCALATION — spalling detection raises the grade")
    print("=" * 78)
    dets = [{"class_id": 1, "class_name": "spalling", "confidence": 0.82,
             "bbox_xyxy": [80, 150, 420, 640]}]
    a_plain = engine.assess(synth(8, 0.6), detections=[])
    a_spall = engine.assess(synth(8, 0.6), detections=dets)
    print(f"without spalling: {a_plain['severity']['grade']}")
    print(f"with spalling   : {a_spall['severity']['grade']}")
    for n in a_spall["severity"]["reasoning"]:
        print(f"    · {n}")

    # --- Real end-to-end run, if the trained models are present -------------
    models_dir = Path(r"D:\nti_project\models")
    det_meta_fp = models_dir / "detection" / "detection_meta.json"
    seg_meta_fp = models_dir / "segmentation" / "segmentation_meta.json"

    if det_meta_fp.exists() and seg_meta_fp.exists() and len(sys.argv) > 1:
        image_path = sys.argv[1]
        print("\n" + "=" * 78)
        print(f"END-TO-END ON {image_path}")
        print("=" * 78)

        sys.path.insert(0, str(Path(__file__).parent))
        # These two helpers are defined in notebooks 05 and 06; in production they
        # live in src/engine/ alongside this file.
        from importlib import import_module
        det_mod = import_module("05_Object_Detection") if False else None  # noqa

        print("Run this through src/backend/main.py instead — the API wires "
              "predict_boxes() and predict_mask() into this engine directly.")
    else:
        print("\nPass an image path as argv[1] with trained models present for an "
              "end-to-end run, or use the /inspect/full endpoint from Notebook 09.")

    print("\n✅ Notebook 08 complete. Copy this file to src/engine/severity.py.")