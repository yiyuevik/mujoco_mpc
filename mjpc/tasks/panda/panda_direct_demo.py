import mujoco
import numpy as np
from mujoco_mpc import direct
import time
import matplotlib.pyplot as plt
import pathlib

# ---------- Load model ----------
model_path = pathlib.Path(__file__).resolve().parent / "xml" / "mjx_scene.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# id of end-effector site
hand_id = model.site("gripper").id

# ---------- MPC solver ----------
T = 50                     # horizon steps
solver = direct.Direct(model, configuration_length=T)

dt = model.opt.timestep
time_array = np.arange(T) * dt  # 改名避免冲突

# ---------- Initial guess ----------
q0 = np.zeros(model.nq)
# linear guess: stay still
q_guess = np.tile(q0[:, None], (1, T))

# Goal position in world frame
goal = np.array([0.3, 0.3, 0.5])

# ---------- Fill Direct buffers ----------
for t in range(T):
    q_t = q_guess[:, t]
    # forward to get predicted ee pos
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_t
    mujoco.mj_forward(model, data)
    hand_pos = data.site_xpos[hand_id]

    # sensor prediction: [ee_pos, joint_positions]
    sensor_pred = np.concatenate([hand_pos, q_t])

    # measurements and masks
    meas = np.zeros(model.nsensordata)
    mask = np.zeros(model.nsensor, dtype=int)

    if t == 0:
        meas[3:] = q0
        mask[1:] = 1  # joints only
    if t == T - 1:
        meas[:3] = goal
        mask[0] = 1  # end-effector position

    solver.data(
        t,
        configuration=q_t,
        sensor_measurement=meas,
        sensor_prediction=sensor_pred,
        sensor_mask=mask,
        force_measurement=np.zeros(model.nv),
        time=np.array([time_array[t]]),
    )

# ---------- settings ----------
solver.settings( max_smoother_iterations=40,    
    max_search_iterations=10,     
    cost_tolerance=1e-6,
    gradient_tolerance=1e-5,
    first_step_position_sensors=True,sensor_flag=True)

print("Optimizing …")
start_time = time.time()
solver.optimize()
end_time = time.time()
print("Done.")
print(f"Optimization took {end_time - start_time:.3f} seconds.")

# ---------- Extract & show ----------
q_opt = np.stack([solver.data(t)["configuration"] for t in range(T)], axis=1)
u_opt = np.stack([solver.data(t)["force_prediction"] for t in range(T)], axis=1)

print("first 5 control torques (Nm):")
print(np.round(u_opt[:, :5], 3))

# ---------- 计算末端执行器位置轨迹 ----------
ee_positions = np.zeros((3, T))  # xyz positions over time
for t in range(T):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = q_opt[:, t]
    mujoco.mj_forward(model, data)
    ee_positions[:, t] = data.site_xpos[hand_id]

# ---------- 绘制轨迹 ----------
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# 绘制关节角度轨迹
axes[0].set_title('Joint Angle Trajectories')
for j in range(model.nq):
    axes[0].plot(time_array, q_opt[j, :], label=f'Joint {j+1}', linewidth=2)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Joint Angle (rad)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 绘制控制力矩轨迹
axes[1].set_title('Control Torque Trajectories')
for j in range(model.nu):
    axes[1].plot(time_array, u_opt[j, :], label=f'Actuator {j+1}', linewidth=2)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Control Torque (Nm)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 绘制末端执行器位置轨迹
axes[2].set_title('End-Effector Position Trajectory')
axes[2].plot(time_array, ee_positions[0, :], label='X', color='red', linewidth=2)
axes[2].plot(time_array, ee_positions[1, :], label='Y', color='green', linewidth=2)
axes[2].plot(time_array, ee_positions[2, :], label='Z', color='blue', linewidth=2)

# 添加目标位置的水平线
axes[2].axhline(y=goal[0], color='red', linestyle='--', alpha=0.7, label='Target X')
axes[2].axhline(y=goal[1], color='green', linestyle='--', alpha=0.7, label='Target Y')
axes[2].axhline(y=goal[2], color='blue', linestyle='--', alpha=0.7, label='Target Z')

axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Position (m)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/panda_trajectory_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------- 计算并显示轨迹统计信息 ----------
print(f"\n=== Trajectory Statistics ===")
print(f"Joint trajectory shape: {q_opt.shape}")  # (nq, T)
print(f"Control trajectory shape: {u_opt.shape}")  # (nu, T)
print(f"End-effector trajectory shape: {ee_positions.shape}")  # (3, T)
print(f"Number of joints: {model.nq}")
print(f"Number of actuators: {model.nu}")
print(f"Trajectory length: {T} steps ({T*dt:.2f} seconds)")

# 显示关节运动范围
print(f"\nJoint motion ranges:")
print("Joint | Min angle | Max angle | Range")
print("-" * 40)
for j in range(model.nq):
    min_angle = np.min(q_opt[j, :])
    max_angle = np.max(q_opt[j, :])
    range_angle = max_angle - min_angle
    print(f"  {j:2d}  | {min_angle:8.3f} | {max_angle:8.3f} | {range_angle:6.3f}")

# 显示末端执行器位置范围
print(f"\nEnd-effector position ranges:")
print("Axis | Min pos   | Max pos   | Range")
print("-" * 35)
axes_names = ['X', 'Y', 'Z']
for i in range(3):
    min_pos = np.min(ee_positions[i, :])
    max_pos = np.max(ee_positions[i, :])
    range_pos = max_pos - min_pos
    print(f"  {axes_names[i]}  | {min_pos:8.3f} | {max_pos:8.3f} | {range_pos:6.3f}")

# Optional: animate with MuJoCo Viewer
try:
    import mujoco.viewer
    mujoco.mj_resetData(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)
    for t in range(T):
        time.sleep(0.01)  # 加快动画速度
        data.qpos[:] = q_opt[:, t]
        mujoco.mj_forward(model, data)
        viewer.sync()
    
    # 只打印最终位置
    final_hand_pos = data.site_xpos[hand_id]
    print(f"\n=== Final Results ===")
    print(f"Target: [{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]")
    print(f"Actual: [{final_hand_pos[0]:.3f}, {final_hand_pos[1]:.3f}, {final_hand_pos[2]:.3f}]")
    print(f"Error:  {np.linalg.norm(final_hand_pos - goal):.3f}")
    
    viewer.close()
except Exception as e:
    print("viewer skipped:", e)

# ---------- 可选：保存关节轨迹到文件 ----------
try:
    import os
    output_dir = "/tmp/panda_trajectory"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存关节轨迹
    np.savetxt(f"{output_dir}/joint_trajectory.txt", q_opt.T, 
               header=f"Joint trajectory (T={T}, nq={model.nq})\nEach row is one time step, columns are joint angles",
               fmt='%.6f')
    
    # 保存控制输入
    np.savetxt(f"{output_dir}/control_trajectory.txt", u_opt.T,
               header=f"Control trajectory (T={T}, nu={model.nu})\nEach row is one time step, columns are control torques",
               fmt='%.6f')
    
    # 保存末端执行器位置轨迹
    np.savetxt(f"{output_dir}/ee_position_trajectory.txt", ee_positions.T,
               header=f"End-effector position trajectory (T={T})\nEach row is one time step, columns are X, Y, Z positions",
               fmt='%.6f')
    
    print(f"\nTrajectory data saved to: {output_dir}/")
    print(f"Plot saved to: /tmp/panda_trajectory_plot.png")
    
except Exception as e:
    print(f"Failed to save trajectory data: {e}")
