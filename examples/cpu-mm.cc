#include <Eigen/Eigen>

#include <cstdlib>
#include <string>
#include <chrono>
#include <ctime>
#include <iostream>

using namespace Eigen;
using namespace std;

typedef MatrixXf M;
typedef VectorXf V;

struct T {
  explicit T(const std::string& x) : x(x), start(std::chrono::high_resolution_clock::now()) {}
  ~T() {
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(end - start);
    cerr << x << " took " << time_span.count() << "s\n";
  }
  void report() { cerr << x << endl; }
  const std::string x;
  std::chrono::high_resolution_clock::time_point start;
};

int main(int argc, char** argv) {
  srand(1);
  const int msize = 128;
  for (unsigned d1 = 8; d1 < 15; ++d1) {
    for (unsigned d2 = 8; d2 < 15; ++d2) {
      const int hsize = pow(2, d1);
      const int xsize = pow(2, d2);
      cerr << "DIM: " << d1 << ' ' << d2 << ' ' << hsize << ' ' << xsize << endl;
      M W = M::Random(hsize,xsize);
      V b = V::Random(hsize);
      M X = M::Random(xsize,msize);
      M Y1 = M::Random(hsize,msize);
      M Y2 = M::Random(hsize,msize);
      M Y3 = M::Random(hsize,msize);
      { T t("Single Matrix-Matrix");
        Y1 = (W * X).colwise() + b; }
      {
        T t("Iterated Matrix-vector");
        for (int m = 0; m < msize; ++m) {
          Y2.col(m) = W * X.col(m) + b;
        }
       }
       {
         T t("C++ for loops");
         // a_ij = a_ik b_kj
         for (int i = 0; i < hsize; ++i)
           for (int j = 0; j < msize; ++j) {
             Y3(i,j) = b(i);
             for (int k = 0; k < xsize; ++k)
               Y3(i,j) += W(i,k) * X(k,j);
           }
        }
      cerr << "Single MM:\n";
      cerr << Y1.block(0,0,4,4) << endl;
      cerr << "Iterated Mv:\n";
      cerr << Y2.block(0,0,4,4) << endl;
      cerr << "For loops:\n";
      cerr << Y3.block(0,0,4,4) << endl;
      cerr << "-----------------------------------------------\n";
    }
  }
}

