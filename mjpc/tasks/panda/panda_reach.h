#pragma once
#include "mjpc/task.h"
#include <array>

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
};

}  // namespace mjpcce mjpc
