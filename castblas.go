package castblas

import "github.com/gonum/blas"

// An Implementation uses a 64-bit BLAS implementation
// except for certain level-3 operations.
type Implementation struct {
	blas.Float64
	DownCast blas.Float32Level3
}

func (i *Implementation) Dgemm(tA, tB blas.Transpose, m, n, k int, alpha float64, a []float64,
	lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	a32 := downcast(a)
	b32 := downcast(b)
	c32 := downcast(c)
	i.DownCast.Sgemm(tA, tB, m, n, k, float32(alpha), a32, lda, b32, ldb, float32(beta),
		c32, ldc)
	upcast(c32, c)
}

func downcast(in []float64) []float32 {
	res := make([]float32, len(in))
	for i, x := range in {
		res[i] = float32(x)
	}
	return res
}

func upcast(in []float32, out []float64) {
	for i, x := range in {
		out[i] = float64(x)
	}
}
