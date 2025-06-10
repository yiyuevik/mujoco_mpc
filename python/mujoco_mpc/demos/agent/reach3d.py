import matplotlib.pyplot as plt
import mediapy as media
import mujoco
from mujoco_mpc import agent as agent_lib
import numpy as np
import pathlib
import os
import time

# %%
# 加载 Panda 机械臂模型
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/panda/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# %%
# 创建 MPC agent (使用你的 Reach3D 任务)
agent = agent_lib.Agent(task_id="Reach3D", model=model)

# 设置成本权重
agent.set_cost_weights({
    "Reach": 1.0,  # 末端执行器位置误差
})
print("Cost weights:", agent.get_cost_weights())

# 先查看可用的任务参数
print("Available task parameters:", agent.get_task_parameters())

# Reach3D 任务使用默认目标位置
target_position = [0.3, 0.3, 0.5]  # 仅用于图表参考
print("使用任务默认目标位置")

print("Final task parameters:", agent.get_task_parameters())

# %%
# 仿真参数
T = 1000  # 仿真步数
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T - 1))
time_array = np.zeros(T)  # 改名为 time_array

# 成本记录
cost_total = np.zeros(T - 1)
cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

# 末端执行器位置记录
end_effector_pos = np.zeros((3, T))

# 每步总时间
step_times = np.zeros(T - 1)  
# MPC 优化时间
mpc_times = np.zeros(T - 1)   
# 物理仿真时间
physics_times = np.zeros(T - 1)  

# %%
# 初始化
mujoco.mj_resetData(model, data)

# 设置初始关节角度（可选）
# data.qpos[:] = [0, -0.5, 0, -2.0, 0, 1.5, 0.785]  # 示例初始姿态
mujoco.mj_forward(model, data)

# 记录初始状态
qpos[:, 0] = data.qpos
qvel[:, 0] = data.qvel
time_array[0] = data.time

# 获取初始末端执行器位置 - 使用 "eeff" 而不是 "hand"
hand_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eeff")
if hand_site_id >= 0:
    end_effector_pos[:, 0] = data.site_xpos[hand_site_id]
    print(f"Found end effector site 'eeff' with id: {hand_site_id}")
else:
    print("Warning: 'eeff' site not found!")

frames = []
FPS = 1.0 / model.opt.timestep

# %%
# 主仿真循环
print("开始 Reach3D 任务仿真...")

for t in range(T - 1):
    step_start = time.perf_counter()
    
    if t % 200 == 0:
        print(f"t = {t}, 末端位置: {end_effector_pos[:, t]}")

    # 更新 MPC 状态
    agent.set_state(
        time=data.time,
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
        mocap_pos=data.mocap_pos,
        mocap_quat=data.mocap_quat,
        userdata=data.userdata,
    )

    # 运行 MPC 优化 - 计时
    mpc_start = time.perf_counter()
    num_steps = 10
    for _ in range(num_steps):
        agent.planner_step()
    mpc_end = time.perf_counter()
    mpc_times[t] = mpc_end - mpc_start

    # 应用控制输入
    data.ctrl = agent.get_action()
    ctrl[:, t] = data.ctrl

    # 记录成本
    cost_total[t] = agent.get_total_cost()
    for i, c in enumerate(agent.get_cost_term_values().items()):
        cost_terms[i, t] = c[1]

    # 物理仿真步进 - 计时
    physics_start = time.perf_counter()
    mujoco.mj_step(model, data)
    physics_end = time.perf_counter()
    physics_times[t] = physics_end - physics_start

    # 记录状态
    qpos[:, t + 1] = data.qpos
    qvel[:, t + 1] = data.qvel
    time_array[t + 1] = data.time
    
    # 记录末端执行器位置
    if hand_site_id >= 0:
        end_effector_pos[:, t + 1] = data.site_xpos[hand_site_id]

    # 渲染
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)
    
    step_end = time.perf_counter()
    step_times[t] = step_end - step_start
    
    # 每100步打印一次时间统计
    if t % 100 == 0 and t > 0:
        print(f"步骤 {t}: 总时间={step_times[t]*1000:.2f}ms, MPC={mpc_times[t]*1000:.2f}ms, 物理={physics_times[t]*1000:.2f}ms")

print("仿真完成!")

# %%
# 保存结果
output_dir = "/tmp/reach3d_results"
os.makedirs(output_dir, exist_ok=True)

# 保存视频
SLOWDOWN = 0.5
media.write_video(f"{output_dir}/reach3d_simulation.mp4", frames, fps=SLOWDOWN * FPS)
print(f"视频已保存到: {output_dir}/reach3d_simulation.mp4")

# %%
# 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 末端执行器轨迹
axes[0, 0].plot(time_array, end_effector_pos[0, :], label="X", color="red")
axes[0, 0].plot(time_array, end_effector_pos[1, :], label="Y", color="green")
axes[0, 0].plot(time_array, end_effector_pos[2, :], label="Z", color="blue")
axes[0, 0].axhline(y=target_position[0], color="red", linestyle="--", alpha=0.5)
axes[0, 0].axhline(y=target_position[1], color="green", linestyle="--", alpha=0.5)
axes[0, 0].axhline(y=target_position[2], color="blue", linestyle="--", alpha=0.5)
axes[0, 0].set_title("末端执行器位置")
axes[0, 0].set_xlabel("时间 (s)")
axes[0, 0].set_ylabel("位置 (m)")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 关节角度
axes[0, 1].plot(time_array, qpos[0, :], label="Joint 1")
axes[0, 1].plot(time_array, qpos[1, :], label="Joint 2")
axes[0, 1].plot(time_array, qpos[2, :], label="Joint 3")
axes[0, 1].set_title("关节角度")
axes[0, 1].set_xlabel("时间 (s)")
axes[0, 1].set_ylabel("角度 (rad)")
axes[0, 1].legend()
axes[0, 1].grid(True)

# 控制输入
axes[1, 0].plot(time_array[:-1], ctrl[0, :], label="Joint 1 Torque")
axes[1, 0].plot(time_array[:-1], ctrl[1, :], label="Joint 2 Torque")
axes[1, 0].plot(time_array[:-1], ctrl[2, :], label="Joint 3 Torque")
axes[1, 0].set_title("控制输入")
axes[1, 0].set_xlabel("时间 (s)")
axes[1, 0].set_ylabel("力矩 (Nm)")
axes[1, 0].legend()
axes[1, 0].grid(True)

# 成本函数
axes[1, 1].plot(time_array[:-1], cost_total, label="总成本", color="black", linewidth=2)
for i, c in enumerate(agent.get_cost_term_values().items()):
    axes[1, 1].plot(time_array[:-1], cost_terms[i, :], label=c[0])
axes[1, 1].set_title("成本函数")
axes[1, 1].set_xlabel("时间 (s)")
axes[1, 1].set_ylabel("成本")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(f"{output_dir}/reach3d_analysis.png", dpi=300)
print(f"分析图已保存到: {output_dir}/reach3d_analysis.png")

# %%
# 计算性能指标
final_position = end_effector_pos[:, -1]
position_error = np.linalg.norm(final_position - np.array(target_position))
print(f"\n=== 性能评估 ===")
print(f"目标位置: {target_position}")
print(f"最终位置: {final_position}")
print(f"位置误差: {position_error:.4f} m")
print(f"总控制成本: {np.sum(cost_total):.4f}")