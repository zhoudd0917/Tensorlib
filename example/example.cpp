#include <iostream>
#include <tensorlib/tensorlib.hpp>

using namespace std;

int main() {
  // 2d vector
  variable x = TensorFactory::create(
      vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                    11.0f, 12.0f},
      vector<size_t>{2, 2, 3}, Device::CPU, true);

  variable y =
      TensorFactory::randn(vector<size_t>{2, 3, 4}, 0., 1., Device::CPU, true);

  variable z = matmul(x, y);
  variable w = relu(z) + z;
  variable l = transpose(w);

  cout << "x: " << x->to_string() << endl;
  cout << "y: " << y->to_string() << endl;
  cout << "z: " << z->to_string() << endl;
  cout << "w: " << w->to_string() << endl;
  cout << "l: " << l->to_string() << endl;

  variable init_grad_w = make_shared<Tensor>(
      vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
      vector<size_t>{2, 4, 2}, Device::CPU);

  l->backward(init_grad_w);

  cout << "x grad: " << x->autograd_meta().grad_->to_string() << endl;
  cout << "y grad: " << y->autograd_meta().grad_->to_string() << endl;
  cout << "z grad: " << z->autograd_meta().grad_->to_string() << endl;
  cout << "w grad: " << w->autograd_meta().grad_->to_string() << endl;
  cout << "l grad: " << l->autograd_meta().grad_->to_string() << endl;
  return 0;
}