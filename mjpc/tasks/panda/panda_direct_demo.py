import mujoco
import numpy as np
from mujoco_mpc import direct

# ---------- Load model ----------
model = mujoco.MjModel.from_xml_path("mjpc/tasks/panda/panda.xml")
data  = mujoco.MjData(model)

# id of end-effector site
hand_id = model.site("hand").id

# ---------- MPC solver ----------
T = 150                     # horizon steps
solver = direct.Direct(model, configuration_length=T)

dt = model.opt.timestep
time = np.arange(T) * dt

# ---------- Initial guess ----------
q0 = model.key_qpos[0] if model.nkey else np.zeros(model.nq)
# linear guess: stay still
q_guess = np.tile(q0[:, None], (1, T))

# Goal position in world frame
goal = np.array([0.3, 0.3, 0.5])

# ---------- Fill Direct buffers ----------
for t in range(T):
    d = solver.data(t)

    # configuration guess
    d["configuration"] = q_guess[:, t]

    # forward to get predicted ee pos
    mujoco.mj_resetData(model, data)
    data.qpos[:] = d["configuration"]
    mujoco.mj_forward(model, data)
    hand_pos = data.site_xpos[hand_id]

    # sensor prediction / measurement
    d["sensor_measurement"] = goal           # what we WANT
    d["sensor_prediction"]  = hand_pos       # what we HAVE
    d["sensor_mask"]        = np.ones(3, dtype=int)

    # forces & time
    d["force_measurement"]  = np.zeros(model.nv)
    d["time"] = time[t]

# ---------- noise & settings ----------
solver.noise(process=1e-4*np.ones(model.nv),
             sensor =1.0 *np.ones(3))
solver.settings(max_search_iterations=50,
                cost_tolerance=1e-6)

print("Optimizing …")
solver.optimize()
print("Done.")

# ---------- Extract & show ----------
q_opt = np.stack([solver.data(t)["configuration"] for t in range(T)], axis=1)
u_opt = np.stack([solver.data(t)["force_prediction"] for t in range(T)], axis=1)

print("first 5 control torques (Nm):")
print(np.round(u_opt[:, :5], 3))

# Optional: animate with MuJoCo Viewer
try:
    import mujoco.viewer
    mujoco.mj_resetData(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)
    for t in range(T):
        data.qpos[:] = q_opt[:, t]
        mujoco.mj_forward(model, data)
        viewer.sync()
    viewer.close()
except Exception as e:
    print("viewer skipped:", e)
