//! A module for all decoding needs.
use crate::error::*;
use crate::{Code, MAX_CODESIZE, MAX_ENTRIES};
use core::iter::zip;
use smallvec::SmallVec;

/// The state for decoding data with an LZW algorithm.
///
/// The same structure can be utilized with streams as well as your own buffers and driver logic.
/// It may even be possible to mix them if you are sufficiently careful not to lose or skip any
/// already decode data in the process.
///
/// This is a sans-IO implementation, meaning that it only contains the state of the decoder and
/// the caller will provide buffers for input and output data when calling the basic
/// [`decode_bytes`] method. Nevertheless, a number of _adapters_ are provided in the `into_*`
/// methods for decoding with a particular style of common IO.
///
/// * [`decode`] for decoding once without any IO-loop.
/// * [`into_async`] for decoding with the `futures` traits for asynchronous IO.
/// * [`into_stream`] for decoding with the standard `io` traits.
/// * [`into_vec`] for in-memory decoding.
///
/// [`decode_bytes`]: #method.decode_bytes
/// [`decode`]: #method.decode
/// [`into_async`]: #method.into_async
/// [`into_stream`]: #method.into_stream
/// [`into_vec`]: #method.into_vec
pub struct Decoder {
    state: DecodeState,
}

#[derive(Clone)]
pub struct Link {
    prev: Code,
    depth: u8,
    byte: u8,
}

#[derive(Default)]
struct CodeBuffer {
    /// A buffer of individual bits. The oldest code is kept in the high-order bits.
    bit_buffer: u64,
    /// A precomputed mask for this code.
    code_mask: u16,
    /// The current code size.
    code_size: u8,
    /// The number of bits in the buffer.
    bits: u8,
}

struct DecodeState {
    /// The original minimum code size.
    min_size: u8,
    /// The table of decoded codes.
    table: Table,
    /// The buffer of decoded data.
    buffer: Buffer,
    /// The link which we are still decoding and its original code.
    last: Option<(Code, Link)>,
    /// The next code entry.
    next_code: Code,
    /// Code to reset all tables.
    clear_code: Code,
    /// Code to signal the end of the stream.
    end_code: Code,
    /// A stored flag if the end code has already appeared.
    has_ended: bool,
    /// If tiff then bumps are a single code sooner.
    is_tiff: bool,
    /// Do we allow stream to start without an explicit reset code?
    implicit_reset: bool,
    /// The buffer for decoded words.
    code_buffer: CodeBuffer,
}

struct Buffer {
    bytes: SmallVec<[u8; 4096]>,
    most_recent_byte: u8,
}

struct Table {
    inner: SmallVec<[Link; 4096]>,
    //min_size: u8,
}

impl Decoder {
    /// Create a new decoder with the specified bit order and symbol size.
    ///
    /// The algorithm for dynamically increasing the code symbol bit width is compatible with the
    /// original specification. In particular you will need to specify an `Lsb` bit oder to decode
    /// the data portion of a compressed `gif` image.
    ///
    /// # Panics
    ///
    /// The `size` needs to be in the interval `0..=12`.
    pub fn new(size: u8) -> Self {
        let state = DecodeState::new(size);

        Decoder { state }
    }

    /// Decode some bytes from `inp` and write result to `out`.
    ///
    /// This will consume a prefix of the input buffer and write decoded output into a prefix of
    /// the output buffer. See the respective fields of the return value for the count of consumed
    /// and written bytes. For the next call You should have adjusted the inputs accordingly.
    ///
    /// The call will try to decode and write as many bytes of output as available. It will be
    /// much more optimized (and avoid intermediate buffering) if it is allowed to write a large
    /// contiguous chunk at once.
    ///
    /// See [`into_stream`] for high-level functions (that are only available with the `std`
    /// feature).
    ///
    /// [`into_stream`]: #method.into_stream
    pub fn decode_bytes(&mut self, inp: &[u8], out: &mut [u8]) -> BufferResult {
        self.state.advance(inp, out)
    }

    /// Check if the decoding has finished.
    ///
    /// No more output is produced beyond the end code that marked the finish of the stream. The
    /// decoder may have read additional bytes, including padding bits beyond the last code word
    /// but also excess bytes provided.
    pub fn has_ended(&self) -> bool {
        self.state.has_ended()
    }

    /// Ignore an end code and continue.
    ///
    /// This will _not_ reset any of the inner code tables and not have the effect of a clear code.
    /// It will instead continue as if the end code had not been present. If no end code has
    /// occurred then this is a no-op.
    ///
    /// You can test if an end code has occurred with [`has_ended`](#method.has_ended).
    /// FIXME: clarify how this interacts with padding introduced after end code.
    #[allow(dead_code)]
    pub(crate) fn restart(&mut self) {
        self.state.restart();
    }

    /// Reset all internal state.
    ///
    /// This produce a decoder as if just constructed with `new` but taking slightly less work. In
    /// particular it will not deallocate any internal allocations. It will also avoid some
    /// duplicate setup work.
    pub fn reset(&mut self) {
        self.state.reset();
    }
}

impl DecodeState {
    fn new(min_size: u8) -> Self {
        DecodeState {
            min_size,
            table: Table::new(min_size),
            buffer: Buffer::new(),
            last: None,
            clear_code: 1 << min_size,
            end_code: (1 << min_size) + 1,
            next_code: (1 << min_size) + 2,
            has_ended: false,
            is_tiff: false,
            implicit_reset: true,
            code_buffer: CodeBuffer::new(min_size),
        }
    }

    fn init_tables(&mut self) {
        self.code_buffer.reset(self.min_size);
        self.next_code = (1 << self.min_size) + 2;
        self.table.init(self.min_size);
    }

    fn reset_tables(&mut self) {
        self.code_buffer.reset(self.min_size);
        self.next_code = (1 << self.min_size) + 2;
        self.table.clear(self.min_size);
    }
}

impl DecodeState {
    fn has_ended(&self) -> bool {
        self.has_ended
    }

    fn restart(&mut self) {
        self.has_ended = false;
    }

    fn reset(&mut self) {
        self.table.init(self.min_size);
        self.last = None;
        self.restart();
        self.code_buffer = CodeBuffer::new(self.min_size);
    }

    fn advance_and_transform<T>(
        &mut self,
        mut inp: &[u8],
        mut out: &mut [T],
        transformer: fn(u8) -> T,
    ) -> BufferResult {
        // Skip everything if there is nothing to do.
        //
        //
        //
        //
        let mut status = Ok(LzwStatus::Ok);
        let mut code_link = None;
        let start_out_size = out.len();
        let start_in_size = out.len();

        match self.last.take() {
            // No last state? This is the first code after a reset?
            None => {
                match self.next_symbol(&mut inp) {
                    // Plainly invalid code.
                    Some(code) if code > self.next_code => status = Err(LzwError::InvalidCode),
                    // next_code would require an actual predecessor.
                    Some(code) if code == self.next_code => status = Err(LzwError::InvalidCode),
                    // No more symbols available and nothing decoded yet.
                    // Assume that we didn't make progress, this may get reset to Done if we read
                    // some bytes from the input.
                    None => status = Ok(LzwStatus::NoProgress),
                    // Handle a valid code.
                    Some(init_code) => {
                        let mut init_tables = false;
                        if init_code == self.clear_code {
                            self.init_tables();
                            init_tables = true;
                        } else if init_code == self.end_code {
                            self.has_ended = true;
                            status = Ok(LzwStatus::Done);
                        } else if self.table.is_empty() {
                            if self.implicit_reset {
                                self.init_tables();
                                init_tables = true;
                            } else {
                                // We require an explicit reset.
                                status = Err(LzwError::InvalidCode);
                            }
                        }

                        if init_tables {
                            // Reconstruct the first code in the buffer.

                            let link = self.table.at(init_code).clone();
                            let first_symbol = if init_code == self.clear_code {
                                self.next_symbol(&mut inp).unwrap()
                            } else {
                                init_code
                            };

                            out = self.buffer.fill_reconstruct_and_transform(
                                &self.table,
                                first_symbol,
                                transformer,
                                out,
                            );

                            code_link = Some((first_symbol, link));
                        }
                    }
                }
            }

            // Move the tracking state to the stack.
            Some(tup) => code_link = Some(tup),
        };

        if !self.buffer.bytes.is_empty() {
            let written = self.buffer.drain_buffer_and_transform(out, transformer);
            out = &mut out[written as usize..];
            if out.is_empty() {
                return BufferResult {
                    consumed_in: start_in_size - inp.len(),
                    consumed_out: start_out_size - out.len(),
                    status,
                };
            }
        }

        while let Some(next_code) = self.code_buffer.next_symbol(&mut inp) {
            let (prev_code, link) = code_link.take().unwrap();
            // Reconstruct the first code in the buffer.
            let link = self
                .table
                .derive(&link, self.buffer.most_recent_byte, prev_code);
            self.next_code += 1;
            if self.next_code == self.code_buffer.max_code() - Code::from(self.is_tiff)
                && self.code_buffer.code_size() < MAX_CODESIZE
            {
                self.bump_code_size();
            }

            match next_code {
                endcode if endcode == self.end_code => {
                    self.has_ended = true;
                    status = Ok(LzwStatus::Done);
                    break;
                }
                resetcode if resetcode == self.clear_code => {
                    self.reset_tables();
                    break;
                }
                _ => {
                    out = self.buffer.fill_reconstruct_and_transform(
                        &self.table,
                        next_code,
                        transformer,
                        out,
                    );

                    code_link = Some((next_code, link));
                }
            }

            if out.is_empty() {
                break;
            }
        }

        self.last = code_link;

        return BufferResult {
            consumed_in: start_in_size - inp.len(),
            consumed_out: start_out_size - out.len(),
            status,
        };
    }

    fn advance(&mut self, inp: &[u8], out: &mut [u8]) -> BufferResult {
        self.advance_and_transform(inp, out, |val| val)
    }
}

impl DecodeState {
    fn next_symbol(&mut self, inp: &mut &[u8]) -> Option<Code> {
        self.code_buffer.next_symbol(inp)
    }

    fn bump_code_size(&mut self) {
        self.code_buffer.bump_code_size()
    }
}

impl CodeBuffer {
    fn new(min_size: u8) -> Self {
        Self {
            code_size: min_size + 1,
            code_mask: (1u16 << (min_size + 1)) - 1,
            bit_buffer: 0,
            bits: 0,
        }
    }

    fn reset(&mut self, min_size: u8) {
        self.code_size = min_size + 1;
        self.code_mask = (1 << self.code_size) - 1;
    }

    fn next_symbol(&mut self, inp: &mut &[u8]) -> Option<Code> {
        if self.bits < self.code_size {
            self.refill_bits(inp);
        }

        self.get_bits()
    }

    fn bump_code_size(&mut self) {
        self.code_size += 1;
        self.code_mask = (self.code_mask << 1) | 1;
    }

    fn refill_bits(&mut self, inp: &mut &[u8]) {
        let wish_count = (64 - self.bits) / 8;
        let mut buffer = [0u8; 8];
        let new_bits = match inp.get(..usize::from(wish_count)) {
            Some(bytes) => {
                buffer[..usize::from(wish_count)].copy_from_slice(bytes);
                *inp = &inp[usize::from(wish_count)..];
                wish_count * 8
            }
            None => {
                let new_bits = inp.len() * 8;
                buffer[..inp.len()].copy_from_slice(inp);
                *inp = &[];
                new_bits as u8
            }
        };
        self.bit_buffer |= u64::from_be_bytes(buffer).swap_bytes() << self.bits;
        self.bits += new_bits;
    }

    fn get_bits(&mut self) -> Option<Code> {
        if self.bits < self.code_size {
            return None;
        }

        let mask = u64::from(self.code_mask);
        let code = self.bit_buffer & mask;
        self.bit_buffer >>= self.code_size;
        self.bits -= self.code_size;
        Some(code as u16)
    }

    fn max_code(&self) -> Code {
        self.code_mask
    }

    fn code_size(&self) -> u8 {
        self.code_size
    }
}

impl Buffer {
    fn new() -> Self {
        Buffer {
            bytes: SmallVec::new(),

            most_recent_byte: 0,
        }
    }

    // Fill the buffer by decoding from the table
    fn drain_buffer_and_transform<T>(
        &mut self,
        writer_buffer: &mut [T],
        transformer: impl Fn(u8) -> T,
    ) -> u16 {
        let src_iter = self.bytes.iter().rev().cloned();
        let dest_iter = writer_buffer.iter_mut();
        let mut num_elems = 0;
        for (src, dest) in zip(src_iter, dest_iter) {
            *dest = transformer(src);
            num_elems += 1;
        }
        self.bytes.truncate(self.bytes.len() - num_elems);
        num_elems as u16
    }

    fn fill_reconstruct_and_transform<'outslice, T>(
        &mut self,
        table: &Table,
        code: Code,
        transformer: impl Fn(u8) -> T,
        writer_buffer: &'outslice mut [T],
    ) -> &'outslice mut [T] {
        let first_link = table.at(code);
        if first_link.depth == u8::MAX || first_link.depth > writer_buffer.len() as u8 {
            self.most_recent_byte = table.buffered_reconstruct(code, &mut self.bytes);
            let drained = self.drain_buffer_and_transform(writer_buffer, transformer) as usize;
            return &mut writer_buffer[drained..];
        } else {
            let mut code_iter = code;
            let table = &table.inner[..=usize::from(code)];
            let mut idx = first_link.depth - 1;

            let mut entry = &table[usize::from(code_iter)];
            while code_iter != 0 {
                //(code, cha) = self.table[k as usize];
                // Note: This could possibly be replaced with an unchecked array access if
                //  - value is asserted to be < self.next_code() in push
                //  - min_size is asserted to be < MAX_CODESIZE
                entry = &table[usize::from(code_iter)];
                code_iter = entry.prev;
                writer_buffer[idx as usize] = transformer(entry.byte);
                idx = idx.saturating_sub(1);
            }
            self.most_recent_byte = entry.byte;
            return &mut writer_buffer[first_link.depth as usize..];
        }
    }
}

impl Table {
    fn new(_min_size: u8) -> Self {
        Table {
            inner: SmallVec::with_capacity(MAX_ENTRIES),
        }
    }

    fn derive(&mut self, from: &Link, byte: u8, prev: Code) -> Link {
        let link = from.derive(byte, prev);
        self.inner.push(link.clone());
        link
    }

    fn clear(&mut self, min_size: u8) {
        let static_count = usize::from(1u16 << u16::from(min_size)) + 2;
        self.inner.truncate(static_count);
    }

    fn init(&mut self, min_size: u8) {
        self.inner.clear();

        for i in 0..(1u16 << u16::from(min_size)) {
            self.inner.push(Link::base(i as u8));
        }
        // Clear code.
        self.inner.push(Link::base(0));

        // End code.
        self.inner.push(Link::base(0));
    }

    fn at(&self, code: Code) -> &Link {
        &self.inner[usize::from(code)]
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn buffered_reconstruct(&self, code: Code, out: &mut SmallVec<[u8; 4096]>) -> u8 {
        let mut code_iter = code;
        let table = &self.inner[..=usize::from(code)];
        let len = code_iter;
        while code_iter != 0 {
            //(code, cha) = self.table[k as usize];
            // Note: This could possibly be replaced with an unchecked array access if
            //  - value is asserted to be < self.next_code() in push
            //  - min_size is asserted to be < MAX_CODESIZE
            let entry = &table[usize::from(code_iter)];
            code_iter = core::cmp::min(len, entry.prev);
            out.push(entry.byte);
        }
        out[0]
    }
}

impl Link {
    fn base(byte: u8) -> Self {
        Link {
            byte,
            prev: 0,
            depth: 1,
        }
    }

    fn derive(&self, byte: u8, prev: Code) -> Self {
        Link {
            byte,
            prev,
            depth: self.depth.saturating_add(1),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::Decoder;
    use super::LzwStatus;
    use weezl::decode::Decoder as WzlDecoder;
    use weezl::encode::Encoder;
    use weezl::BitOrder;

    fn make_decoded() -> Vec<u8> {
        const FILE: &'static [u8] =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"));
        return Vec::from(FILE);
    }

    #[test]
    fn darlin_test() {
        let mut encoder = Encoder::new(BitOrder::Lsb, 8);
        let mut base_decoder = WzlDecoder::new(BitOrder::Lsb, 8);
        let encoded: Vec<u8> = "HELLO MY DARLIN HELLO MY RAGTIME GAL".into();
        let out_data = encoder.encode(&encoded).unwrap();
        let value = base_decoder.decode(&out_data).unwrap();
        let mut array_vec = vec![0; encoded.len()];
        let mut decoder = Decoder::new(8);
        let result = decoder.decode_bytes(&out_data[..], &mut array_vec[..]);
        let status = result.status.unwrap();
        assert_eq!(encoded, array_vec);
        assert_eq!(result.consumed_out, array_vec.len());
        assert_eq!(status, LzwStatus::Done);
    }
}
