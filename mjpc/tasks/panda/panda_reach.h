#pragma once
#include "mjpc/task.h"
#include <array>
#include <vector>

namespace mjpc {

class PandaReach : public Task {
 public:
  std::string Name()    const override;
  std::string XmlPath() const override;

  struct ResidualFn : public BaseResidualFn {
    explicit ResidualFn(const PandaReach* t);
    void Residual(const mjModel*, const mjData*, double* r) const override;
   private:
    const PandaReach* task_;
  };

  PandaReach();

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override;
  ResidualFn* InternalResidual() override;
  void TransitionLocked(mjModel*, mjData*) override;

 private:
  ResidualFn residual_;
  std::array<double, 3> goal_{0.3, 0.3, 0.5};   // 目标位置
  
  // 初始猜测相关变量
  std::vector<double> initial_guess_;  // 初始猜测轨迹
  int horizon_;  // 预测时域长度
  double dt_;    // 时间步长
  
  // 多模态相关变量
  std::vector<std::array<double, 3>> initial_positions_;  // 多个初始位置
  int current_mode_;  // 当前使用的模态
  double mode_switch_time_;  // 模态切换时间
  bool mode_switched_;  // 是否已切换模态
  
  // 设置初始猜测的方法
  void SetInitialGuess(const mjModel* m, const mjData* d);
  
  // 辅助函数
  void InitializeModes();
  void UpdateMode(const mjModel* m, const mjData* d);
};

}  // namespace mjpc
