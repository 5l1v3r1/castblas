package castblas

import (
	"runtime"
	"sync"

	"github.com/gonum/blas"
)

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
	var c32 []float32
	if beta != 0 {
		c32 = downcast(c)
	} else {
		c32 = make([]float32, len(c))
	}
	i.DownCast.Sgemm(tA, tB, m, n, k, float32(alpha), a32, lda, b32, ldb, float32(beta),
		c32, ldc)
	upcast(c32, c)
}

func downcast(in []float64) []float32 {
	numRoutines := runtime.GOMAXPROCS(0)
	chunkSize := len(in) / numRoutines
	res := make([]float32, len(in))
	var wg sync.WaitGroup
	for i := 0; i < numRoutines; i++ {
		wg.Add(1)
		startIdx := i * chunkSize
		endIdx := (i + 1) * chunkSize
		if i == numRoutines-1 {
			endIdx = len(in)
		}
		go func() {
			for j := startIdx; j < endIdx; j++ {
				res[j] = float32(in[j])
			}
			wg.Done()
		}()
	}
	wg.Wait()
	return res
}

func upcast(in []float32, out []float64) {
	numRoutines := runtime.GOMAXPROCS(0)
	chunkSize := len(in) / numRoutines
	var wg sync.WaitGroup
	for i := 0; i < numRoutines; i++ {
		wg.Add(1)
		startIdx := i * chunkSize
		endIdx := (i + 1) * chunkSize
		if i == numRoutines-1 {
			endIdx = len(in)
		}
		go func() {
			for j := startIdx; j < endIdx; j++ {
				out[j] = float64(in[j])
			}
			wg.Done()
		}()
	}
	wg.Wait()
}
