// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! Minimal .npy reader (little-endian u8 / f32 / i8 / bool / i64, C order) for fixtures.

use std::fs;
use std::path::Path;

/// One array.
pub struct Npy {
    /// Shape.
    pub shape: Vec<usize>,
    /// numpy dtype string.
    pub dtype: String,
    /// Raw little-endian data.
    pub data: Vec<u8>,
}

impl Npy {
    /// Read a file.
    pub fn load(p: &Path) -> std::io::Result<Npy> {
        let b = fs::read(p)?;
        let bad = || std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad npy {}", p.display()));
        if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
            return Err(bad());
        }
        let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
        let hdr = std::str::from_utf8(&b[off..off + hl]).map_err(|_| bad())?;
        let descr = hdr.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).ok_or_else(bad)?.to_string();
        if hdr.contains("'fortran_order': True") {
            return Err(bad());
        }
        let sh = hdr.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).ok_or_else(bad)?;
        let shape = sh.split(',').filter_map(|t| t.trim().parse::<usize>().ok()).collect();
        Ok(Npy { shape, dtype: descr, data: b[off + hl..].to_vec() })
    }

    /// Values as f32.
    pub fn f32(&self) -> Vec<f32> {
        match self.dtype.as_str() {
            "<f4" => self.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "<f8" => self.data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            "|u1" => self.data.iter().map(|&v| v as f32).collect(),
            "|b1" => self.data.iter().map(|&v| v as f32).collect(),
            "|i1" => self.data.iter().map(|&v| v as i8 as f32).collect(),
            "<i8" => self.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            d => panic!("dtype {d}"),
        }
    }

    /// Raw u8 data.
    pub fn u8(&self) -> Vec<u8> {
        assert_eq!(self.dtype, "|u1");
        self.data.clone()
    }
}
