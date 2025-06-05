#include "mjpc/tasks/panda/panda_reach.h"
#include "mjpc/utilities.h"
#include <mujoco/mujoco.h>

namespace mjpc {

// ---------- public ----------
std::string PandaReach::Name()    const { return "Reach3D"; }
std::string PandaReach::XmlPath() const { return GetModelPath("panda/task.xml"); }

// ---------- residual ----------
PandaReach::ResidualFn::ResidualFn(const PandaReach* t)
    : BaseResidualFn(static_cast<const Task*>(t)), task_(t) {}

void PandaReach::ResidualFn::Residual(const mjModel* m, const mjData* d,
                                      double* r) const {
  // 获取当前末端执行器位置
  const double* hand = SensorByName(m, d, "hand");
  
  // 计算到目标位置的误差
  mju_sub3(r, hand, task_->goal_.data());
  
 
}

// ---------- ctor & hooks ----------
PandaReach::PandaReach() : residual_(this) {
}

void PandaReach::TransitionLocked(mjModel* m, mjData* d) {
  const double* hand = SensorByName(m, d, "hand");
  printf("Hand position: %.3f %.3f %.3f\n", hand[0], hand[1], hand[2]);
}

std::unique_ptr<mjpc::ResidualFn> PandaReach::ResidualLocked() const {
  return std::make_unique<ResidualFn>(this);
}

PandaReach::ResidualFn* PandaReach::InternalResidual() { return &residual_; }

}  // namespace mjpc
