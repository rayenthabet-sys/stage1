"""
main.py — FastAPI backend for olive grove mapping demo.
Run with:  uvicorn main:app --reload --port 8000

Stage 1: Detection  — spatial query of EZZAYRA parcels inside drawn polygon
                       (production: U-Net on Sentinel-2 tiles)
Stage 2: Classification — 3-class model (extensif / intensif / hyper_intensif)
"""

import json, logging, math, time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline import (
    train_model, classify_parcel, detect_parcels_in_zone,
    CLASS_NAMES, centroid, area_ha_shoelace
)
from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

if settings.OPENED_CLIENT_ID:
    log.info("Environment variables loaded successfully.")
else:
    log.warning("OPENED_CLIENT_ID not found in .env or environment.")

# ─── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Olive Grove Mapping API",
    description="Cartographie intelligente des oliveraies tunisiennes — Hack The Harvest 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = Path(__file__).parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

# ─── Global model state ──────────────────────────────────────────────────────

_model      = None   # geo-only model (live inference)
_full_model = None   # geo + spectral model (demo metric)
_metrics    = None
_all_parcels: List[dict] = []   # all EZZAYRA parcels (ext + int + HI)


@app.on_event("startup")
def startup():
    global _model, _full_model, _metrics, _all_parcels
    log.info("Training 3-class model on EZZAYRA dataset …")
    t0 = time.time()
    _model, _full_model, _metrics = train_model()
    log.info(
        "Model ready in %.2f s — CV F1-macro: %.3f (geo) / %.3f (full)",
        time.time() - t0,
        _metrics["geo_cv_f1_macro_mean"],
        _metrics["full_cv_f1_macro_mean"],
    )

    # Pre-load EZZAYRA parcels (extensif + hyper_intensif ground truth)
    data_dir = Path(__file__).parent / "data"
    for fname, systeme in [
        ("parcelles_OlivierExtensif.json",   "extensif"),
        ("parcellesOliviersIntensifs.json",  "hyper_intensif"),
    ]:
        with open(data_dir / fname) as f:
            raw = json.load(f)
        for p in raw["parcels"]:
            p["systeme_label"] = systeme
            _all_parcels.append(p)
    log.info("Loaded %d EZZAYRA parcels", len(_all_parcels))


# ─── Schemas ────────────────────────────────────────────────────────────────

class Coordinate(BaseModel):
    lat: float
    lng: float


# ── Stage 2 only (legacy / direct classification) ──
class ParcelInput(BaseModel):
    id:          Optional[str]  = None
    coordinates: List[Coordinate]
    area_ha:     Optional[float] = None


class DirectClassifyRequest(BaseModel):
    parcelles: List[ParcelInput] = Field(..., min_items=1)


# ── Stage 1 + 2 (main flow: user draws a zone) ──
class CartographierRequest(BaseModel):
    """
    Main endpoint payload.
    polygone_perimetre: GeoJSON-like polygon drawn by the user (zone to analyse).
    date: acquisition date hint for Sentinel-2 (YYYY-MM-DD). Optional for demo.
    """
    polygone_perimetre: List[Coordinate] = Field(
        ..., description="Polygon vertices drawn on the map (lat/lng).", min_items=3
    )
    date: Optional[str] = Field(None, description="Target acquisition date YYYY-MM-DD.")


class OliveraieResult(BaseModel):
    id:           str
    systeme:      str
    confiance:    float
    surface_ha:   float
    ndvi_mean:    float
    ndwi:         float
    ndre:         float
    compactness:  float
    centroid_lat: float
    centroid_lng: float
    coordinates:  List[Coordinate]
    name:         Optional[str] = None


class StatsResult(BaseModel):
    total:              int
    surface_totale_ha:  float
    surface_moyenne_ha: float
    repartition:        dict


class CartographierResponse(BaseModel):
    oliveraies: List[OliveraieResult]
    stats:      StatsResult
    latence_ms: int
    date:       Optional[str] = None
    zone_area_km2: Optional[float] = None


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    html = FRONTEND / "index.html"
    if html.exists():
        return FileResponse(str(html))
    return {"message": "Olive Grove Mapping API — see /docs"}


@app.get("/api/health")
def health():
    return {
        "status":      "ok",
        "model_ready": _model is not None,
        "classes":     CLASS_NAMES,
        "n_parcels":   len(_all_parcels),
        "geo_f1_macro": _metrics["geo_cv_f1_macro_mean"] if _metrics else None,
        "full_f1_macro": _metrics["full_cv_f1_macro_mean"] if _metrics else None,
    }


@app.get("/api/metrics")
def metrics():
    """Return model training metrics and feature importances."""
    if not _metrics:
        raise HTTPException(503, "Model not ready")
    return _metrics


@app.get("/api/parcelles")
def list_parcelles(systeme: Optional[str] = None):
    """Return EZZAYRA parcels as GeoJSON FeatureCollection."""
    parcels = _all_parcels
    if systeme:
        parcels = [p for p in parcels if p["systeme_label"] == systeme]

    features = []
    for p in parcels:
        lat, lng = centroid(p["coordinates"])
        features.append({
            "type": "Feature",
            "properties": {
                "id":      p["id"],
                "name":    p["name"],
                "systeme": p["systeme_label"],
                "area_ha": round(p["area_ha"], 2),
                "centroid_lat": round(lat, 6),
                "centroid_lng": round(lng, 6),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[c["lng"], c["lat"]] for c in p["coordinates"]]
                    + [[p["coordinates"][0]["lng"], p["coordinates"][0]["lat"]]]
                ],
            },
        })
    return {
        "type":     "FeatureCollection",
        "features": features,
        "total":    len(features),
    }


@app.post("/api/cartographier", response_model=CartographierResponse)
def cartographier(req: CartographierRequest):
    """
    Main pipeline endpoint.

    Stage 1 — Detection: find all olive grove parcels inside the drawn polygon.
    Stage 2 — Classification: classify each into extensif / intensif / hyper_intensif.

    POST body:
        {
          "polygone_perimetre": [ {"lat": …, "lng": …}, … ],
          "date": "2026-06-15"           // optional
        }
    """
    if _model is None:
        raise HTTPException(503, "Model not ready — retry in a few seconds.")

    t0 = time.time()

    # ── Stage 1: Detection ───────────────────────────────────────────────────
    zone_coords = [{"lat": c.lat, "lng": c.lng} for c in req.polygone_perimetre]

    # Compute approximate zone area (km²)
    zone_area_ha = area_ha_shoelace(zone_coords)
    zone_area_km2 = round(zone_area_ha / 100, 1)

    detected = detect_parcels_in_zone(zone_coords, _all_parcels)

    if not detected:
        return CartographierResponse(
            oliveraies=[],
            stats=StatsResult(
                total=0,
                surface_totale_ha=0.0,
                surface_moyenne_ha=0.0,
                repartition={c: 0 for c in CLASS_NAMES},
            ),
            latence_ms=int((time.time() - t0) * 1000),
            date=req.date,
            zone_area_km2=zone_area_km2,
        )

    # ── Stage 2: Classification ───────────────────────────────────────────────
    results = []
    for p in detected:
        parcel_dict = {
            "id":          p["id"],
            "coordinates": p["coordinates"],
            "area_ha":     p["area_ha"],
        }
        prediction = classify_parcel(_model, parcel_dict)
        lat, lng   = centroid(p["coordinates"])

        results.append(OliveraieResult(
            id=p["id"],
            name=p.get("name"),
            systeme=prediction["systeme"],
            confiance=prediction["confiance"],
            surface_ha=prediction["surface_ha"],
            ndvi_mean=prediction["ndvi_mean"],
            ndwi=prediction["ndwi"],
            ndre=prediction["ndre"],
            compactness=prediction["compactness"],
            centroid_lat=round(lat, 6),
            centroid_lng=round(lng, 6),
            coordinates=[Coordinate(lat=c["lat"], lng=c["lng"]) for c in p["coordinates"]],
        ))

    # ── Stats ─────────────────────────────────────────────────────────────────
    repartition   = {c: 0 for c in CLASS_NAMES}
    surface_totale = 0.0
    for r in results:
        repartition[r.systeme] += 1
        surface_totale += r.surface_ha

    latence_ms = int((time.time() - t0) * 1000)
    log.info(
        "Zone %.1f km² — detected %d parcels, classified in %d ms",
        zone_area_km2, len(results), latence_ms,
    )

    return CartographierResponse(
        oliveraies=results,
        stats=StatsResult(
            total=len(results),
            surface_totale_ha=round(surface_totale, 2),
            surface_moyenne_ha=round(surface_totale / len(results), 2) if results else 0.0,
            repartition=repartition,
        ),
        latence_ms=latence_ms,
        date=req.date,
        zone_area_km2=zone_area_km2,
    )


@app.post("/api/cartographier/geojson")
def cartographier_geojson(req: CartographierRequest):
    """Same as /api/cartographier but returns a GeoJSON FeatureCollection."""
    result = cartographier(req)
    COLORS = {
        "extensif":       "#1D9E75",
        "intensif":       "#EF9F27",
        "hyper_intensif": "#E24B4A",
    }
    features = []
    for r in result.oliveraies:
        features.append({
            "type": "Feature",
            "properties": {
                "id":         r.id,
                "name":       r.name,
                "systeme":    r.systeme,
                "confiance":  r.confiance,
                "surface_ha": r.surface_ha,
                "ndvi_mean":  r.ndvi_mean,
                "ndwi":       r.ndwi,
                "ndre":       r.ndre,
                "color":      COLORS.get(r.systeme, "#888"),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[c.lng, c.lat] for c in r.coordinates]
                    + [[r.coordinates[0].lng, r.coordinates[0].lat]]
                ],
            },
        })
    return {
        "type":     "FeatureCollection",
        "features": features,
        "stats":    result.stats.model_dump(),
        "date":     result.date,
        "zone_area_km2": result.zone_area_km2,
    }