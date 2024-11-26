#include <iostream>
#include <tensorlib/tensorlib.hpp>

using namespace std;

int main() {
  Device device = Device::GPU;
  // 2d vector
  variable x = TensorFactory::create(
      vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                    11.0f, 12.0f},
      vector<size_t>{2, 2, 3}, device, true);

  variable y = TensorFactory::create(
      vector<float>{2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,
                    10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f,
                    18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, -2.f},
      vector<size_t>{2, 3, 4}, device, true);

  variable z = matmul(x, y);
  variable w = z + z;
  variable l = w + w;

  cout << "x: " << x->to_string() << endl;
  cout << "y: " << y->to_string() << endl;
  cout << "z: " << z->to_string() << endl;
  cout << "w: " << w->to_string() << endl;
  cout << "l: " << l->to_string() << endl;

  variable init_grad_w = make_shared<Tensor>(
      vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f},
      vector<size_t>{2, 2, 4}, device);

  l->backward(init_grad_w);

  cout << "l grad: " << l->autograd_meta().grad_->to_string() << endl;
  cout << "z grad: " << z->autograd_meta().grad_->to_string() << endl;
  cout << "w grad: " << w->autograd_meta().grad_->to_string() << endl;
  cout << "y grad: " << y->autograd_meta().grad_->to_string() << endl;
  cout << "x grad: " << x->autograd_meta().grad_->to_string() << endl;
  return 0;
}