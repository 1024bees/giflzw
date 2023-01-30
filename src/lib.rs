#![cfg_attr(not(test), no_std)]

mod decoder;
mod error;

pub(crate) const MAX_CODESIZE: u8 = 12;
pub(crate) const MAX_ENTRIES: usize = 1 << MAX_CODESIZE as usize;

/// Alias for a LZW code point
pub(crate) type Code = u16;

pub use decoder::Decoder;
