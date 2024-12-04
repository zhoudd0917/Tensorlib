#ifndef GRAD_MODE_HPP
#define GRAD_MODE_HPP

class GradMode {
 public:
  /**
   * Check if gradient computation is currently enabled.
   * @return `true` if gradient computation is enabled, `false` otherwise.
   */
  static bool is_enabled();

  /**
   * Enable or disable gradient computation.
   * @param enabled `true` to enable gradient computation, `false` to disable
   * it.
   */
  static void set_grad_mode(bool enabled);

 private:
  static bool grad_mode_enabled_;
};

// Disables gradient computation when constructed and restores the previous
class NoGradScope {
 public:
  NoGradScope();
  ~NoGradScope();

 private:
  bool prev_mode_;
};

#endif