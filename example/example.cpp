#include <iostream>
#include <tensorlib/tensorlib.hpp>

using namespace std;

int main() {
  Device device = Device::CPU;
  // 2d vector
  variable x = TensorFactory::create(
      vector<float>{1.0f, 2.0f, 3.0f, 10.0f, 11.0f, 12.0f},
      vector<size_t>{2, 1, 3}, device, true);

  variable y =
      TensorFactory::create(vector<float>{1.f, -2.f, -5.f, 3.f, 2.f, 1.f},
                            vector<size_t>{2, 3, 1}, device, true);

  variable z = matmul(x, y);
  variable w = broadcast_to(z, vector<size_t>{2, 3, 5});
  variable l = w * 2.0f;

  cout << "x: " << x->to_string() << endl;
  cout << "y: " << y->to_string() << endl;
  cout << "z: " << z->to_string() << endl;
  cout << "w: " << w->to_string() << endl;
  cout << "l: " << l->to_string() << endl;

  //   variable init_grad_w = make_shared<Tensor>(
  //       vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
  //       vector<size_t>{2, 4}, device);
  variable init_grad_l =
      TensorFactory::randn(l->shape(), 0.0f, 1.0f, device, false);

  l->backward(init_grad_l);

  cout << "l grad: " << l->autograd_meta().grad_->to_string() << endl;
  cout << "w grad: " << w->autograd_meta().grad_->to_string() << endl;
  cout << "z grad: " << z->autograd_meta().grad_->to_string() << endl;
  cout << "y grad: " << y->autograd_meta().grad_->to_string() << endl;
  cout << "x grad: " << x->autograd_meta().grad_->to_string() << endl;
  return 0;
}