/// The result of a coding operation on a pair of buffer.
#[must_use = "Contains a status with potential error information"]
pub struct BufferResult {
    /// The number of bytes consumed from the input buffer.
    pub consumed_in: usize,
    /// The number of bytes written into the output buffer.
    pub consumed_out: usize,
    /// The status after returning from the write call.
    pub status: Result<LzwStatus, LzwError>,
}

/// The status after successful coding of an LZW stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LzwStatus {
    /// Everything went well.
    Ok,
    /// No bytes were read or written and no internal state advanced.
    ///
    /// If this is returned but your application can not provide more input data then decoding is
    /// definitely stuck for good and it should stop trying and report some error of its own. In
    /// other situations this may be used as a signal to refill an internal buffer.
    NoProgress,
    /// No more data will be produced because an end marker was reached.
    Done,
}

/// The error kind after unsuccessful coding of an LZW stream.
#[derive(Debug, Clone, Copy)]
pub enum LzwError {
    /// The input contained an invalid code.
    ///
    /// For decompression this refers to a code larger than those currently known through the prior
    /// decoding stages. For compression this refers to a byte that has no code representation due
    /// to being larger than permitted by the `size` parameter given to the Encoder.
    InvalidCode,
}

impl core::fmt::Display for LzwError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            LzwError::InvalidCode => f.write_str("invalid code in LZW stream"),
        }
    }
}
