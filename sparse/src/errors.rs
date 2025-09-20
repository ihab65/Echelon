use thiserror::Error;

#[derive(Debug, Error)]
pub enum CsrError {
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("Duplicate entry in input")]
    DuplicateEntry,
}

pub type Result<T> = std::result::Result<T, CsrError>;