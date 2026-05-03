#![forbid(unsafe_code)]

//! Knob-grid parsing.
//!
//! The CLI accepts two parameters that drive the Cartesian product:
//!
//! - `--q-grid 5,10,15,...,95` — a comma-separated list of integer
//!   qualities. Codec-specific quality semantics are resolved per-codec
//!   in [`super::encode`]; the grid value is always the generic 0..=100
//!   integer the user thinks in.
//! - `--knob-grid '<json-object>'` — a JSON object mapping knob name to
//!   a list of values. The Cartesian product of those lists, plus the
//!   q-grid, produces the cells encoded for each source image. The set
//!   of valid knob names is per-codec — see [`super::encode::codec_knobs`].
//!
//! ```text
//! --q-grid 25,50,75,90
//! --knob-grid '{"method": [4,5,6], "segments": [1,4]}'
//! ```
//! produces 4 × 3 × 2 = 24 cells per image for the WebP codec.
//!
//! The serialized form preserved into the output TSV is a stable JSON
//! object with sorted keys — the same string is reproducible across
//! runs and easy to filter with `awk` / `jq`.

use serde_json::{Map, Value};
use std::error::Error;
use std::fmt;

/// One cell's knob assignment, e.g. `{"method": 5, "segments": 4}`.
#[derive(Debug, Clone)]
pub struct KnobTuple(pub Map<String, Value>);

impl KnobTuple {
    /// Stable canonical JSON encoding (keys sorted) for TSV output.
    pub fn to_canonical_json(&self) -> String {
        // serde_json::Map preserves insertion order with `preserve_order`,
        // but we want a deterministic key order independent of how the
        // Cartesian iterator built it. Sort keys, then re-emit.
        let mut keys: Vec<&String> = self.0.keys().collect();
        keys.sort();
        let mut sorted = Map::new();
        for k in keys {
            if let Some(v) = self.0.get(k) {
                sorted.insert(k.clone(), v.clone());
            }
        }
        Value::Object(sorted).to_string()
    }
}

/// A knob-grid spec: knob name → list of values. The Cartesian product is
/// expanded by [`KnobGrid::iter_tuples`].
#[derive(Debug, Clone)]
pub struct KnobGrid {
    /// Each entry is a knob name and the ordered list of values it takes
    /// across the sweep. We hold the order the user supplied (for
    /// human-readable diagnostics) but emit canonical-sorted JSON in the
    /// TSV.
    pub axes: Vec<(String, Vec<Value>)>,
}

impl KnobGrid {
    /// An empty grid yields exactly one tuple (the empty assignment), which
    /// keeps the per-image cell count at `q_grid.len()` for callers who
    /// only want to vary quality.
    pub fn empty() -> Self {
        Self { axes: Vec::new() }
    }

    /// Total cells in the Cartesian product. `1` for an empty grid.
    pub fn cell_count(&self) -> usize {
        self.axes
            .iter()
            .map(|(_, v)| v.len())
            .product::<usize>()
            .max(1)
    }

    /// Walk every (knob_name → value) tuple in row-major order.
    pub fn iter_tuples(&self) -> Box<dyn Iterator<Item = KnobTuple> + '_> {
        if self.axes.is_empty() {
            return Box::new(std::iter::once(KnobTuple(Map::new())));
        }
        // We expand iteratively to avoid mut-borrow lifetime issues with a
        // recursive iterator — the cell count is always small (a few dozen).
        let mut tuples: Vec<KnobTuple> = vec![KnobTuple(Map::new())];
        for (name, values) in &self.axes {
            let mut next = Vec::with_capacity(tuples.len() * values.len());
            for t in &tuples {
                for v in values {
                    let mut m = t.0.clone();
                    m.insert(name.clone(), v.clone());
                    next.push(KnobTuple(m));
                }
            }
            tuples = next;
        }
        Box::new(tuples.into_iter())
    }
}

/// Parse `--q-grid 5,10,15` into `[5, 10, 15]`. Whitespace around items is
/// trimmed; empty fields are an error.
pub fn parse_q_grid(s: &str) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            return Err(format!("--q-grid contains an empty field in {s:?}").into());
        }
        let v: u32 = p
            .parse()
            .map_err(|e| format!("invalid q value {p:?}: {e}"))?;
        if v > 100 {
            return Err(format!("--q-grid value {v} out of range (expected 0..=100)").into());
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err("--q-grid is empty".into());
    }
    Ok(out)
}

/// Parse `--knob-grid '<json-object>'`. Empty input produces an empty grid.
pub fn parse_knob_grid(s: &str) -> Result<KnobGrid, Box<dyn Error>> {
    if s.trim().is_empty() {
        return Ok(KnobGrid::empty());
    }
    let v: Value =
        serde_json::from_str(s).map_err(|e| format!("--knob-grid is not valid JSON: {e}"))?;
    let obj = v
        .as_object()
        .ok_or("--knob-grid must be a JSON object {axis: [values]}")?;
    let mut axes: Vec<(String, Vec<Value>)> = Vec::new();
    for (k, v) in obj.iter() {
        let arr = v
            .as_array()
            .ok_or_else(|| KnobGridError(format!("knob {k:?} must map to an array of values")))?;
        if arr.is_empty() {
            return Err(format!("knob {k:?} has an empty value list").into());
        }
        axes.push((k.clone(), arr.clone()));
    }
    Ok(KnobGrid { axes })
}

/// Lightweight error for grid parsing.
#[derive(Debug)]
struct KnobGridError(String);

impl fmt::Display for KnobGridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for KnobGridError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_grid_parses() {
        assert_eq!(parse_q_grid("25,50,75,90").unwrap(), vec![25, 50, 75, 90]);
    }

    #[test]
    fn q_grid_rejects_empty() {
        assert!(parse_q_grid("").is_err());
        assert!(parse_q_grid("25,,50").is_err());
    }

    #[test]
    fn q_grid_rejects_out_of_range() {
        assert!(parse_q_grid("25,200").is_err());
    }

    #[test]
    fn knob_grid_empty_input_is_empty_grid() {
        let g = parse_knob_grid("").unwrap();
        assert_eq!(g.cell_count(), 1);
        let tuples: Vec<_> = g.iter_tuples().collect();
        assert_eq!(tuples.len(), 1);
        assert!(tuples[0].0.is_empty());
    }

    #[test]
    fn knob_grid_cartesian_expand() {
        let g = parse_knob_grid(r#"{"method": [4, 5, 6], "segments": [1, 4]}"#).unwrap();
        assert_eq!(g.cell_count(), 6);
        let tuples: Vec<_> = g.iter_tuples().collect();
        assert_eq!(tuples.len(), 6);
        // Each tuple has both keys.
        for t in &tuples {
            assert!(t.0.contains_key("method"));
            assert!(t.0.contains_key("segments"));
        }
    }

    #[test]
    fn knob_grid_canonical_json_is_sorted() {
        let g = parse_knob_grid(r#"{"zaxis": [1], "aaxis": [2]}"#).unwrap();
        let t = g.iter_tuples().next().unwrap();
        let json = t.to_canonical_json();
        // "aaxis" should appear before "zaxis".
        let aa = json.find("aaxis").unwrap();
        let za = json.find("zaxis").unwrap();
        assert!(aa < za, "expected canonical (sorted) keys, got {json}");
    }
}
