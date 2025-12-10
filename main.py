import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g_acc = 9.81
init_pos = 1.0
init_vel = 10.0

def exact_fn(t):
    return init_pos + init_vel * t - 0.5 * g_acc * t ** 2

t_start, t_end = 0.0, 2.0
num_samples = 10
time_samples = np.linspace(t_start, t_end, num_samples)

np.random.seed(0)
noise_std = 0.7
pos_exact = exact_fn(time_samples)
pos_noisy = pos_exact + noise_std * np.random.randn(num_samples)

time_tensor = torch.tensor(time_samples, dtype=torch.float32).view(-1, 1)
pos_tensor = torch.tensor(pos_noisy, dtype=torch.float32).view(-1, 1)

time_for_plot = np.linspace(t_start, t_end, 100).reshape(-1, 1).astype(np.float32)
time_for_plot_tensor = torch.tensor(time_for_plot, requires_grad=False)

def calc_derivatives(output, input_var):
    return torch.autograd.grad(output, input_var, grad_outputs=torch.ones_like(output), create_graph=True)[0]

class SimpleNN(nn.Module):
    def __init__(self, hidden_units=20):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )
    def forward(self, x):
        return self.layers(x)

model_phys = SimpleNN(hidden_units=20)
model_plain = SimpleNN(hidden_units=20)

def loss_data(model, t_in, y_true):
    y_pred = model(t_in)
    return torch.mean((y_pred - y_true) ** 2)

def loss_physics(model, t_in):
    t_in.requires_grad = True
    y_pred = model(t_in)
    dy_dt = calc_derivatives(y_pred, t_in)
    physics_val = init_vel - g_acc * t_in
    return torch.mean((dy_dt - physics_val) ** 2)

def loss_initial(model):
    t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    y0_pred = model(t0)
    return torch.mean((y0_pred - init_pos) ** 2)

optimizer_model_phys = torch.optim.Adam(model_phys.parameters(), lr=0.01)
optimizer_model_plain = torch.optim.Adam(model_plain.parameters(), lr=0.01)

weight_data_loss = 2.0
weight_phys_loss = 2.0
weight_init_loss = 2.0

epochs = 2000
display_step = 200
animation_step = 50

phys_predictions_anim = []
plain_predictions_anim = []
animation_epochs = []

model_phys.train()
model_plain.train()

for epoch in range(epochs):
    optimizer_model_phys.zero_grad()
    optimizer_model_plain.zero_grad()

    loss_d = loss_data(model_phys, time_tensor, pos_tensor)
    loss_p = loss_physics(model_phys, time_tensor)
    loss_i = loss_initial(model_phys)

    loss_plain = loss_data(model_plain, time_tensor, pos_tensor)

    total_phys_loss = weight_data_loss * loss_d + weight_phys_loss * loss_p + weight_init_loss * loss_i

    total_phys_loss.backward()
    loss_plain.backward()

    optimizer_model_phys.step()
    optimizer_model_plain.step()

    if (epoch + 1) % animation_step == 0:
        model_phys.eval()
        model_plain.eval()
        with torch.no_grad():
            pred_phys_anim = model_phys(time_for_plot_tensor).detach().numpy()
            pred_plain_anim = model_plain(time_for_plot_tensor).detach().numpy()
            phys_predictions_anim.append(pred_phys_anim)
            plain_predictions_anim.append(pred_plain_anim)
            animation_epochs.append(epoch + 1)
        model_phys.train()
        model_plain.train()

    if (epoch + 1) % display_step == 0:
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Plain NN Loss = {loss_plain.item():.6f}, "
              f"Total Physics NN Loss = {total_phys_loss.item():.6f}, "
              f"Data Loss = {loss_d.item():.6f}, "
              f"Physics Loss = {loss_p.item():.6f}, "
              f"Init Loss = {loss_i.item():.6f}")

model_phys.eval()
model_plain.eval()

pred_plot_phys = model_phys(time_for_plot_tensor).detach().numpy()
pred_plot_plain = model_plain(time_for_plot_tensor).detach().numpy()
exact_plot = exact_fn(time_for_plot)

plt.figure(figsize=(10, 5))
plt.plot(time_for_plot, exact_plot, label='Exact', color='blue', linestyle=':')
plt.plot(time_for_plot, pred_plot_phys, label='Physics NN', color='red')
plt.plot(time_for_plot, pred_plot_plain, label='Plain NN', color='green')
plt.scatter(time_samples, pos_tensor.numpy(), label='Noisy Samples', color='black', marker='x')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.show()

fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
ax_anim.plot(time_for_plot, exact_plot, label='Exact', color='blue', linestyle=':')
ax_anim.scatter(time_samples, pos_tensor.numpy(), label='Noisy Samples', color='black', marker='x')
line_phys, = ax_anim.plot([], [], lw=2, label='Physics NN', color='red')
line_plain, = ax_anim.plot([], [], lw=2, label='Plain NN', color='green')
epoch_txt = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)

ax_anim.set_xlabel('Time')
ax_anim.set_ylabel('Position')
ax_anim.legend(loc='upper right')
ax_anim.set_title('Training Progress')

def anim_init():
    line_phys.set_data([], [])
    line_plain.set_data([], [])
    epoch_txt.set_text('')
    return line_phys, line_plain, epoch_txt

def anim_update(i):
    line_phys.set_data(time_for_plot, phys_predictions_anim[i])
    line_plain.set_data(time_for_plot, plain_predictions_anim[i])
    epoch_txt.set_text(f'Epoch: {animation_epochs[i]}')
    return line_phys, line_plain, epoch_txt

animation_obj = animation.FuncAnimation(fig_anim, anim_update, frames=len(animation_epochs),
                                        init_func=anim_init, blit=True, interval=200)

try:
    animation_obj.save('training_animation.mp4', writer='ffmpeg', fps=10)
    print("Animation saved as training_animation.mp4")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure ffmpeg is installed and in your PATH.")

plt.show()
