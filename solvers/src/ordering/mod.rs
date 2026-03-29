pub mod graph;
pub mod permutation;
pub mod amd;
pub mod rcm;

pub use graph::Graph;
pub use permutation::Permutation;
pub use amd::amd;
pub use rcm::rcm;

use sparse::{SparseScalar, SymCsrMatrix};

// -----------------------------------------------------------------
// Ordering strategy enum
// -----------------------------------------------------------------

/// Fill-reduction ordering strategy for sparse Cholesky factorization.
///
/// Pass this to [`SparseSolver::set_ordering`](crate::cholesky::SparseSolver::set_ordering)
/// before calling `analyze`.  The default strategy when `set_ordering` is not
/// called is [`Ordering::Rcm`].
///
/// ## Choosing a strategy
///
/// | Strategy        | Best for                                               |
/// |-----------------|--------------------------------------------------------|
/// | [`Rcm`]         | 1-D and regular 2-D meshes with natural bandwidth      |
/// | [`Amd`]         | Irregular meshes, frame joints, star-like connectivity |
/// | [`Natural`]     | Already-optimal orderings; debugging; tiny systems     |
/// | [`Custom`]      | Pre-computed orderings from an external tool           |
///
/// When in doubt, prefer [`Amd`] — it consistently produces equal or less
/// fill than [`Rcm`] across the problem types encountered in structural FEM.
///
/// [`Rcm`]: Ordering::Rcm
/// [`Amd`]: Ordering::Amd
/// [`Natural`]: Ordering::Natural
/// [`Custom`]: Ordering::Custom
#[derive(Debug, Clone)]
pub enum Ordering {
    /// **Reverse Cuthill-McKee** — graph-bandwidth reduction.
    ///
    /// Performs a BFS from a pseudo-peripheral node, enqueueing neighbours in
    /// ascending-degree order, then reverses the result.  O(n + nnz).
    ///
    /// Best when the input graph is a path, grid, or other structure where
    /// the natural node numbering already gives compact bandwidth.
    Rcm,

    /// **Approximate Minimum Degree** — greedy fill minimisation.
    ///
    /// At each step eliminates the node with minimum (approximate) degree and
    /// updates the quotient graph.  O(n · d²) where d is the elimination
    /// degree, which is small for sparse FEM graphs.
    ///
    /// Best for irregular meshes, frame structures with shared joints, and
    /// any graph where the hub-and-spoke pattern would cause RCM to create
    /// wide cliques.
    Amd,

    /// **Natural ordering** — no reordering, identity permutation.
    ///
    /// The matrix is factorized exactly as assembled.  Use when:
    /// - The DOF numbering is already fill-optimal (e.g. from a nested
    ///   dissection renumbering by the mesh generator).
    /// - Comparing orderings for benchmarking or debugging.
    /// - The system is tiny and ordering overhead exceeds its benefit.
    Natural,

    /// **Custom permutation** — caller supplies the reordering.
    ///
    /// The permutation must satisfy `perm[new] = old` (new index → old index)
    /// and be a bijection of `0..n`.  Use when integrating with an external
    /// ordering tool (e.g. METIS, nested dissection from a mesh library).
    Custom(Permutation),
}

impl Ordering {
    /// Compute the permutation for a given matrix.
    ///
    /// Consumes `self` for the `Custom` variant (moves the permutation out
    /// without cloning).  All other variants compute a fresh permutation.
    pub(crate) fn into_permutation<T>(self, k: &SymCsrMatrix<T>) -> Permutation 
        where T: SparseScalar
    {
        match self {
            Ordering::Rcm => {
                let g = Graph::from_sym(k);
                rcm(&g)
            }
            Ordering::Amd    => amd(k),
            Ordering::Natural => Permutation::identity(k.n),
            Ordering::Custom(p) => p,
        }
    }
}

impl Default for Ordering {
    /// The default ordering is RCM, matching historical behaviour of the
    /// solver before the `Ordering` enum was introduced.
    fn default() -> Self {
        Ordering::Rcm
    }
}