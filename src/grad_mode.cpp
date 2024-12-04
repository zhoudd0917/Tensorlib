#include <tensorlib/grad_mode.hpp>

bool GradMode::grad_mode_enabled_ = true;

bool GradMode::is_enabled() { return grad_mode_enabled_; }

void GradMode::set_grad_mode(bool enabled) { grad_mode_enabled_ = enabled; }

NoGradScope::NoGradScope() : prev_mode_(GradMode::is_enabled()) {
  GradMode::set_grad_mode(false);
}

NoGradScope::~NoGradScope() { GradMode::set_grad_mode(prev_mode_); }