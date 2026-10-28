"""Reading the spectra the pipeline fits.

The pipeline's one data assumption is that every source is a single FITS file
holding three arrays: wavelength, flux and error.  This module finds those three
arrays -- in a binary table as columns, or in an image HDU with a matching
``EXTNAME`` -- converts the wavelength to Angstroms when the header says it is in
something else, removes the Galactic extinction the header declares, de-redshifts
to the rest frame, and reports what it did.

Two of those corrections are read from the file rather than invented: the
redshift (``[data] z_keys``) and the colour excess (``[data] e_bv_keys``, an
``A_V`` being divided by ``R_V``).  A file that declares neither is fitted as it
stands -- rest frame, no extinction -- but the report says so instead of leaving
the reader to assume the correction was applied.

Nothing else is derived: a missing error array is an error, not a guess, because
a fitted uncertainty that silently came from ``std(flux)`` is not a measurement.
"""

from __future__ import annotations

import csv
import glob as glob_module
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import PipelineError
from .schema import PipelineConfig

__all__ = ["SourceRef", "Spectrum", "resolve_sources", "read_spectrum",
           "flux_statistics"]


@dataclass(frozen=True)
class SourceRef:
    """One spectrum to fit, before it has been read."""

    source_id: str
    path: str
    redshift: Optional[float] = None


@dataclass(frozen=True)
class Spectrum:
    """A spectrum, on the rest-frame wavelength grid the model works in."""

    source_id: str
    path: str
    wave: np.ndarray                  # rest frame, Angstrom
    flux: np.ndarray
    error: np.ndarray
    redshift: float
    z_source: str                     # 'header:Z' | 'catalog' | 'z_default'
    flux_unit: Optional[str]
    wave_unit: Optional[str]
    e_bv: float = 0.0                 # colour excess removed from flux and error
    e_bv_source: str = "assumed"      # 'header:E(B-V)' | 'header:A_V' | 'assumed' | 'none'
    extinction_rv: float = 3.1
    #: ``(velocity_km_s, flux)`` from ``[data] narrow_template_key``, or ``None``
    #: when the file does not carry a template.  Read here rather than in the
    #: builder so the template travels with the spectrum it belongs to.
    narrow_template: Optional[Tuple[np.ndarray, np.ndarray]] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def n_pixels(self) -> int:
        return int(self.wave.size)

    def summary(self) -> Dict[str, Any]:
        """The JSON-able description written into ``run_report.json``."""
        return {
            "source_id": self.source_id,
            "path": self.path,
            "n_pixels": self.n_pixels,
            "wave_range": [float(self.wave.min()), float(self.wave.max())],
            "redshift": self.redshift,
            "z_source": self.z_source,
            "flux_unit": self.flux_unit,
            "wave_unit": self.wave_unit,
            "e_bv": self.e_bv,
            "e_bv_source": self.e_bv_source,
            "extinction_rv": self.extinction_rv,
            "narrow_template": (None if self.narrow_template is None else {
                "points": int(self.narrow_template[0].size),
                "velocity_range": [float(self.narrow_template[0].min()),
                                   float(self.narrow_template[0].max())],
            }),
            "extensions": dict(self.provenance),
        }


# --------------------------------------------------------------------------------------
# Which spectra are there
# --------------------------------------------------------------------------------------
def _wildcarded(pattern: Optional[str]) -> bool:
    return bool(pattern) and any(char in pattern for char in "*?[")


def _source_id_for(path: Path) -> str:
    name = Path(path).name
    for suffix in (".fits.gz", ".fits", ".fit.gz", ".fit", ".fz"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def resolve_sources(config: PipelineConfig) -> List[SourceRef]:
    """Expand the ``[data]`` section into the list of spectra to fit."""
    data = config.data
    base = Path(config.path).parent
    if data.source == "file":
        path = Path(data.file)
        if not path.is_file():
            raise PipelineError(f"spectrum not found: {path}")
        return [SourceRef(source_id=_source_id_for(path), path=str(path))]

    if data.source == "files":
        refs = []
        for name in data.files:
            path = Path(name)
            if not path.is_file():
                raise PipelineError(f"spectrum not found: {path}")
            refs.append(SourceRef(source_id=_source_id_for(path), path=str(path)))
        return refs

    if data.source == "glob":
        pattern = str(data.glob)
        if not os.path.isabs(pattern):
            pattern = str(base / pattern)
        matches = sorted(p for p in glob_module.glob(pattern) if os.path.isfile(p))
        if not matches:
            raise PipelineError(f"glob matched no files: {data.glob!r} "
                                f"(looked for {pattern})")
        return [SourceRef(source_id=_source_id_for(Path(p)), path=str(Path(p).resolve()))
                for p in matches]

    if data.source == "catalog":
        return _read_catalog(config)

    raise PipelineError(f"unknown data source {data.source!r}")   # pragma: no cover


def _read_catalog(config: PipelineConfig) -> List[SourceRef]:
    data = config.data
    path = Path(data.catalog)
    if not path.is_file():
        raise PipelineError(f"catalog not found: {path}")
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        raise PipelineError(f"catalog is empty: {path}")
    reader = csv.DictReader(rows, delimiter=delimiter,
                            skipinitialspace=True)
    fields = [name.strip() for name in (reader.fieldnames or [])]
    by_lower = {name.lower(): name for name in fields}
    file_column = by_lower.get(data.catalog_file_column.lower())
    if file_column is None:
        raise PipelineError(
            f"catalog {path} has no column {data.catalog_file_column!r}; it has "
            f"{', '.join(fields) or 'no columns'}")
    id_column = by_lower.get(data.catalog_id_column.lower())
    z_column = (by_lower.get(data.catalog_z_column.lower())
                if data.catalog_z_column else None)

    refs: List[SourceRef] = []
    for line_number, row in enumerate(reader, start=2):
        name = (row.get(file_column) or "").strip()
        if not name:
            continue
        spectrum = Path(name).expanduser()
        if not spectrum.is_absolute():
            spectrum = path.parent / spectrum
        if not spectrum.is_file():
            raise PipelineError(f"catalog {path} line {line_number}: "
                                f"spectrum not found: {spectrum}")
        redshift = None
        if z_column is not None:
            raw = (row.get(z_column) or "").strip()
            if raw:
                try:
                    redshift = float(raw)
                except ValueError:
                    raise PipelineError(
                        f"catalog {path} line {line_number}: {z_column} = {raw!r} "
                        "is not a number") from None
        source_id = ""
        if id_column is not None:
            source_id = (row.get(id_column) or "").strip()
        refs.append(SourceRef(source_id=source_id or _source_id_for(spectrum),
                              path=str(spectrum.resolve()), redshift=redshift))
    if not refs:
        raise PipelineError(f"catalog has no rows: {path}")
    ids = [ref.source_id for ref in refs]
    duplicates = sorted({name for name in ids if ids.count(name) > 1})
    if duplicates:
        raise PipelineError(f"catalog {path} gives the same id to more than one "
                            f"source: {', '.join(duplicates)}")
    return refs


# --------------------------------------------------------------------------------------
# Reading one spectrum
# --------------------------------------------------------------------------------------
def _hdu_label(index: int, hdu) -> str:
    name = hdu.header.get("EXTNAME") or hdu.header.get("EXTNMAE") or ""
    return f"{index}:{name or 'PRIMARY' if index == 0 else name or 'HDU' + str(index)}"


def _table_columns(hdu) -> List[str]:
    try:
        return [str(name) for name in hdu.columns.names]
    except Exception:                            # pragma: no cover - defensive
        return []


def _is_table(hdu) -> bool:
    try:
        from astropy.io import fits
    except ImportError:                          # pragma: no cover - astropy required
        raise PipelineError("reading FITS spectra needs astropy") from None
    return isinstance(hdu, (fits.BinTableHDU, fits.TableHDU))


def _is_image(hdu) -> bool:
    from astropy.io import fits
    return isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) \
        or (isinstance(hdu, fits.hdu.base._ValidHDU) and hdu.data is not None
            and not _is_table(hdu))


def _array_from_hdu(hdu, key: str) -> Tuple[np.ndarray, Optional[str]]:
    """Extract ``key`` from one HDU: a table column or an image array."""
    if _is_table(hdu):
        names = _table_columns(hdu)
        for name in names:
            if name.upper() == key.upper():
                values = np.asarray(hdu.data[name], dtype=float)
                try:
                    unit = hdu.columns[name].unit
                except Exception:                # pragma: no cover - defensive
                    unit = None
                return values.ravel(), unit
        return None, None
    if hdu.data is not None:
        name = (hdu.header.get("EXTNAME") or "").strip()
        if name and name.upper() == key.upper():
            values = np.asarray(hdu.data, dtype=float)
            unit = (hdu.header.get("BUNIT") or hdu.header.get("CUNIT1")
                    or hdu.header.get("UNIT"))
            return values.ravel(), unit
    return None, None


def _locate(hdul, key: str, hint: Optional[str], role: str):
    """Find ``key`` in the file, honouring an optional HDU hint."""
    from astropy.io import fits

    indices = range(len(hdul))
    if hint is not None:
        hint_text = str(hint)
        found = None
        if hint_text.lstrip("-").isdigit():
            index = int(hint_text)
            if not -len(hdul) <= index < len(hdul):
                raise PipelineError(f"{role}: HDU index {index} is out of range "
                                    f"(the file has {len(hdul)} HDUs)")
            found = [index % len(hdul)]
        else:
            found = [index for index, hdu in enumerate(hdul)
                     if (hdu.header.get("EXTNAME") or "").strip().upper()
                     == hint_text.strip().upper()]
            if not found:
                raise PipelineError(f"{role}: no HDU named {hint_text!r}; "
                                    f"the file has {_describe_hdus(hdul)}")
        indices = found

    for index in indices:
        values, unit = _array_from_hdu(hdul[index], key)
        if values is not None:
            return values, unit, hdul[index].header, index
    where = (f"HDU {hint!r}" if hint is not None else "any HDU")
    raise PipelineError(f"{role}: no array named {key!r} in {where}; "
                        f"the file has {_describe_hdus(hdul)}")


def _describe_hdus(hdul) -> str:
    described = []
    for index, hdu in enumerate(hdul):
        if _is_table(hdu):
            described.append(f"[{index}] table {hdu.header.get('EXTNAME', '')!r} "
                             f"columns {_table_columns(hdu)}")
        elif hdu.data is not None:
            described.append(f"[{index}] image "
                             f"{(hdu.header.get('EXTNAME') or '(unnamed)')!r} "
                             f"shape {np.shape(hdu.data)}")
        else:
            described.append(f"[{index}] {type(hdu).__name__}")
    return "; ".join(described) or "no HDUs"


#: Names a header may use for "this axis is in Angstroms".
_ANGSTROM = {"a", "aa", "ang", "angs", "angstrom", "angstroms", "å", "ångström"}

#: Names a header may use for "this axis is a velocity in km/s".
_KM_PER_S = {"km/s", "km s-1", "km s^-1", "kms-1", "km/s (rest)"}


def _to_km_per_s(values: np.ndarray, unit: Optional[str], role: str) -> np.ndarray:
    """Convert a template's velocity column to km/s.

    A column with no unit is taken to be km/s already: the template is described
    in ``config_fit.rst`` as a velocity in km/s, and a table written by hand
    usually omits the unit.
    """
    if unit is None or not str(unit).strip():
        return values
    label = str(unit).strip()
    if label.lower().replace(" ", "") in {name.replace(" ", "") for name in _KM_PER_S}:
        return values
    try:
        import astropy.units as u
    except ImportError:                          # pragma: no cover - astropy required
        raise PipelineError("converting velocity units needs astropy") from None
    try:
        quantity = u.Unit(label)
    except Exception:
        raise PipelineError(f"{role}: {label!r} is not a unit astropy understands") from None
    if not quantity.is_equivalent(u.km / u.s):
        raise PipelineError(
            f"{role}: unit {label!r} is not a velocity; the template's first column "
            "must be a velocity convertible to km/s")
    return values * float((1.0 * quantity).to_value(u.km / u.s))


def _read_narrow_template(hdul, key: str, path) -> Tuple[np.ndarray, np.ndarray]:
    """The ``(velocity_km_s, flux)`` table named by ``[data] narrow_template_key``.

    The table is read by position -- first column velocity, second flux -- because
    that is the contract in the documentation, and it lets a file written by any
    tool (including one that names its columns ``V``/``F``) be used as it stands.
    """
    wanted = str(key).strip()
    index: Optional[int] = None
    if wanted.lstrip("-").isdigit():
        position = int(wanted)
        if -len(hdul) <= position < len(hdul):
            index = position % len(hdul)
    else:
        for candidate, hdu in enumerate(hdul):
            if (hdu.header.get("EXTNAME") or "").strip().upper() == wanted.upper():
                index = candidate
                break
    if index is None:
        raise PipelineError(
            f"{path}: [data] narrow_template_key = {key!r} names no HDU; "
            f"the file has {_describe_hdus(hdul)}")
    hdu = hdul[index]
    if not _is_table(hdu):
        raise PipelineError(
            f"{path}: [data] narrow_template_key = {key!r} is an "
            f"{type(hdu).__name__}, not a table; the template must be a two-column "
            "table, velocity (km/s) then normalised flux")
    names = _table_columns(hdu)
    if len(names) < 2:
        raise PipelineError(
            f"{path}: template HDU {key!r} has {len(names)} column(s) "
            f"({', '.join(names) or 'none'}); need velocity and flux")
    velocity = np.asarray(hdu.data[names[0]], dtype=float).ravel()
    flux = np.asarray(hdu.data[names[1]], dtype=float).ravel()
    try:
        unit = hdu.columns[names[0]].unit
    except Exception:                            # pragma: no cover - defensive
        unit = None
    velocity = _to_km_per_s(velocity, unit, f"{path}: template column {names[0]!r}")
    if velocity.size != flux.size:
        raise PipelineError(
            f"{path}: template HDU {key!r} has {velocity.size} velocities but "
            f"{flux.size} fluxes")
    if velocity.size < 3:
        raise PipelineError(
            f"{path}: template HDU {key!r} has {velocity.size} row(s); a profile "
            "needs at least 3")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(flux)):
        raise PipelineError(f"{path}: template HDU {key!r} has non-finite values")
    order = np.argsort(velocity, kind="stable")
    if not np.array_equal(order, np.arange(velocity.size)):
        velocity, flux = velocity[order], flux[order]
    if np.any(np.diff(velocity) <= 0.0):
        raise PipelineError(
            f"{path}: template HDU {key!r} repeats a velocity ({names[0]!r} column); "
            "the profile must be single-valued in velocity")
    return velocity, flux


def _to_angstrom(values: np.ndarray, unit: Optional[str], role: str
                 ) -> Tuple[np.ndarray, str]:
    """Convert a wavelength array to Angstroms; returns (values, unit label)."""
    if unit is None or not str(unit).strip():
        return values, "Angstrom (assumed)"
    label = str(unit).strip()
    if label.lower() in _ANGSTROM:
        return values, "Angstrom"
    try:
        import astropy.units as u
    except ImportError:                          # pragma: no cover - astropy required
        raise PipelineError("converting wavelength units needs astropy") from None
    try:
        quantity = u.Unit(label)
    except Exception:
        raise PipelineError(f"{role}: {label!r} is not a unit astropy understands; "
                            "give [data] wave_unit to say what the axis is in") from None
    if not quantity.is_equivalent(u.AA):
        raise PipelineError(
            f"{role}: wavelength unit {label!r} is not a length; the model works "
            "on a wavelength grid, so the axis must be convertible to Angstrom "
            "(set [data] wave_unit if the header does not say)")
    factor = float((1.0 * quantity).to_value(u.AA))
    return values * factor, label


def _header_redshift(headers: Sequence, keys: Sequence[str]
                     ) -> Tuple[Optional[float], Optional[str]]:
    for header in headers:
        for key in keys:
            if key in header:
                try:
                    value = float(header[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and -0.99 < value:
                    return value, f"header:{key}"
    return None, None


#: Header keys that hold a colour excess rather than an extinction; an ``A_V``
#: is divided by ``R_V`` to become one.  The distinction matters for the label
#: the report prints, not for the arithmetic.
_COLOUR_KEYS = frozenset({"E(B-V)", "EBV", "E(BV)", "EB-V", "EBV_GAL"})
_EXTINCTION_KEYS = frozenset({"A_V", "AV", "AV_GAL", "A(V)"})


def _header_extinction(headers: Sequence, keys: Sequence[str], r_v: float
                       ) -> Tuple[Optional[float], Optional[str]]:
    """``E(B-V)`` declared by a header, and which key supplied it."""
    for header in headers:
        for key in keys:
            if key not in header:
                continue
            try:
                value = float(header[key])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value) or value < 0.0:
                continue
            if key.upper() in _COLOUR_KEYS:
                return value, f"header:{key}"
            if key.upper() in _EXTINCTION_KEYS:
                return value / r_v, f"header:{key} (A_V / R_V)"
            # an unlabelled key in [data] e_bv_keys is read as a colour excess
            return value, f"header:{key}"
    return None, None


def _extinction_factor(wave: np.ndarray, e_bv: float, r_v: float) -> Optional[np.ndarray]:
    """``10**(0.4 A_lambda)``: what multiplies the flux to remove the extinction.

    ``wave`` is the *observed* grid, which is what a reddening law is tabulated
    against.  ``None`` means the package cannot do it (no ``extinction``), and the
    caller then reports the correction as not applied rather than pretending.
    """
    try:
        from extinction import ccm89
    except ImportError:                            # pragma: no cover - dependency
        return None
    return 10.0 ** (0.4 * np.asarray(ccm89(wave, e_bv, r_v=r_v, unit="aa"), dtype=float))


def read_spectrum(config: PipelineConfig, ref: SourceRef) -> Spectrum:
    """Read one spectrum onto the rest-frame grid, applying ``[data]`` cuts.

    Raises
    ------
    PipelineError
        If an array is missing, the lengths disagree, the wavelength unit is not
        a length, or the cuts leave fewer than ``min_pixels`` pixels.
    """
    try:
        from astropy.io import fits
    except ImportError:                          # pragma: no cover - astropy required
        raise PipelineError("reading FITS spectra needs astropy") from None

    data = config.data
    path = Path(ref.path)
    try:
        with fits.open(path, memmap=False) as hdul:
            wave, wave_unit, wave_header, wave_index = _locate(
                hdul, data.wave_key, data.wave_hdu, "wavelength")
            flux, flux_unit, flux_header, flux_index = _locate(
                hdul, data.flux_key, data.flux_hdu, "flux")
            error, _, _, error_index = _locate(
                hdul, data.error_key, data.error_hdu, "error")
            headers = [flux_header, hdul[0].header]
            z_header, z_header_label = _header_redshift(headers, data.z_keys)
            e_bv_header, e_bv_label = _header_extinction(
                headers, data.e_bv_keys, data.extinction_rv)
            # read while the file is still open: the template belongs to this
            # spectrum, so it is carried on the Spectrum rather than re-opened
            template = (None if data.narrow_template_key is None
                        else _read_narrow_template(hdul, data.narrow_template_key, path))
    except PipelineError:
        raise
    except OSError as exc:
        raise PipelineError(f"cannot open {path}: {exc}") from None

    if not (wave.size == flux.size == error.size):
        raise PipelineError(
            f"{path}: wavelength, flux and error have different lengths "
            f"({wave.size}, {flux.size}, {error.size})")
    if wave.size == 0:
        raise PipelineError(f"{path}: the arrays are empty")

    wave_unit_label = data.wave_unit or wave_unit
    wave, wave_unit_label = _to_angstrom(wave, wave_unit_label, "wavelength")
    flux_unit_label = data.flux_unit or flux_unit
    if not np.all(np.isfinite(wave)):
        raise PipelineError(f"{path}: the wavelength array is not finite")
    if np.any(np.diff(wave) == 0):
        raise PipelineError(f"{path}: the wavelength array has repeated values")

    # what the source says about its own redshift, in order of explicitness:
    # the catalog row (chosen by the user for this source), then the header,
    # then [data] z_default
    if ref.redshift is not None:
        redshift, z_source = float(ref.redshift), "catalog"
    elif z_header is not None:
        redshift, z_source = z_header, z_header_label or "header"
    else:
        redshift, z_source = data.z_default, "z_default"
    if redshift <= -1.0:
        raise PipelineError(f"{path}: redshift {redshift} is not usable "
                            "(must be greater than -1)")

    # Galactic extinction, removed on the observed grid before the frame is changed.
    # Both flux and error are multiplied: dereddening is a multiplicative correction,
    # so leaving the error alone would silently change the weights of the fit.
    if data.extinction:
        if e_bv_header is None:
            e_bv, e_bv_source = 0.0, "assumed"
            warnings.warn(
                f"{path}: no Galactic colour excess in the header "
                f"({'/'.join(data.e_bv_keys)}); fitting the flux as observed "
                "(set [data] extinction = false to silence this)", RuntimeWarning)
        else:
            e_bv, e_bv_source = float(e_bv_header), e_bv_label or "header"
        factor = _extinction_factor(wave, e_bv, data.extinction_rv) if e_bv else None
        if e_bv and factor is None:                # pragma: no cover - dependency
            warnings.warn(f"{path}: astropy's extinction package is missing; "
                          "E(B-V) not removed", RuntimeWarning)
            e_bv, e_bv_source = 0.0, "skipped"
        elif factor is not None and e_bv:
            flux = flux * factor
            error = error * factor
    else:
        e_bv, e_bv_source = 0.0, "none"

    if z_source == "z_default" and not data.z_default_given:
        warnings.warn(
            f"{path}: no redshift in the header or catalog; using "
            f"[data] z_default = {data.z_default}", RuntimeWarning)

    wave_rest = wave / (1.0 + redshift)
    keep = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    n_dropped = int(np.count_nonzero(~keep))
    if data.wave_min is not None:
        keep &= wave_rest >= data.wave_min
    if data.wave_max is not None:
        keep &= wave_rest <= data.wave_max
    for low, high in data.mask:
        keep &= ~((wave_rest >= low) & (wave_rest <= high))
    if not np.all(keep):
        wave_rest, flux, error = wave_rest[keep], flux[keep], error[keep]
    if flux.size == 0:
        raise PipelineError(f"{path}: the cuts in [data] remove every pixel")
    if flux.size < data.min_pixels:
        raise PipelineError(
            f"{path}: {flux.size} pixels left after the cuts, fewer than "
            f"[data] min_pixels = {data.min_pixels}")

    order = np.argsort(wave_rest, kind="stable")
    if not np.array_equal(order, np.arange(wave_rest.size)):
        wave_rest, flux, error = wave_rest[order], flux[order], error[order]

    provenance = {
        "wavelength": _hdu_role(wave_index, data.wave_key),
        "flux": _hdu_role(flux_index, data.flux_key),
        "error": _hdu_role(error_index, data.error_key),
        "pixels_dropped": n_dropped,
        "sorted": bool(not np.array_equal(order, np.arange(order.size))),
    }
    if data.extinction:
        provenance["extinction"] = e_bv_source
    if data.narrow_template_key is not None:
        provenance["narrow_template"] = (f"HDU {data.narrow_template_key} "
                                        f"({template[0].size} rows)")
    return Spectrum(source_id=ref.source_id, path=str(path), wave=wave_rest,
                    flux=flux, error=error, redshift=float(redshift),
                    z_source=z_source, flux_unit=flux_unit_label,
                    wave_unit=wave_unit_label, e_bv=float(e_bv),
                    e_bv_source=e_bv_source,
                    extinction_rv=float(data.extinction_rv),
                    narrow_template=template,
                    provenance=provenance)


def _hdu_role(index: int, key: str) -> str:
    """A short human label such as ``'HDU 1 column WAVELENGTH'``."""
    return f"HDU {index} ({key})"


def flux_statistics(spectrum: Spectrum, continuum_percentile: float
                    ) -> Dict[str, float]:
    """The per-source numbers :mod:`galspec.pipeline.schema` expressions use."""
    from .schema import source_namespace

    return source_namespace(spectrum.flux, spectrum.wave, spectrum.redshift,
                            continuum_percentile)