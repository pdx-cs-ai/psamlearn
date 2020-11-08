pub struct EvenChunks<'a, T> {
    blocksize: usize,
    rem: usize,
    rest: &'a[T],
}

impl<'a, T> EvenChunks<'a, T> {
    pub fn nchunks(slice: &'a[T], nchunks: usize) -> Self {
        let nslice = slice.len();
        let mut blocksize = nslice / nchunks;
        let rem = nslice - blocksize * nchunks;
        if rem > 0 {
            blocksize += 1;
        }
        EvenChunks {
            blocksize,
            rem,
            rest: slice,
        }
    }
}

impl<'a, T> Iterator for EvenChunks<'a, T> {
    type Item = &'a[T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            return None;
        }
        let blocksize = if self.rem == 1 {
            self.blocksize -= 1;
            self.blocksize + 1
        } else {
            self.blocksize
        };
        assert!(self.rest.len() >= blocksize);
        if self.rem > 0 {
            self.rem -= 1;
        }
        let result = &self.rest[..blocksize];
        self.rest = &self.rest[blocksize..];
        Some(result)
    }
}
