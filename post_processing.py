"""Post-processing code for EVGen paper here: https://arxiv.org/abs/2108.03762
Note that data is not made publically available due to privacy concerns. Code can be used
with your own dataset to achieve similar results."""
import torch
from matplotlib import pyplot as plt
import h5py
from torch.autograd import Variable
import numpy as np
import os

FONT_SIZE = 14
DPI = 400
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'figure.dpi': 400})
plt.rcParams.update({'lines.linewidth': 2.5})
LEGEND_FONT_SIZE = 12

# ---- Inputs -----
# training_id = "AWS_fast_v7"
# model_number = 189750
# training_id = "AWS_fast_v8"
# model_number = 282000
# training_id = "AWS_fast_v9"
# model_number = 339000
training_id = "3-17-21_discrete_SCWGAN_v2"
model_number = 350000
noise_dim = 80
c_dim = 8
project_path = ""
model_type = "d"    # d: discrete, c: continuous.
DPI = 400
# ------------------

PLOT_COLORS = ["#581845", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


model_path = "Results/{}/models/model_{}".format(training_id, model_number)
train_data_path = "Results/{}/data_train.h5".format(training_id)
test_data_path = "Results/{}/data_test.h5".format(training_id)

# Load models and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = torch.load(model_path, map_location=device)
D = models["D_model"].to(device)
G = models["G_model"].to(device)
with h5py.File(train_data_path, 'r') as hf:
    data_train = hf['data_train'][:]
with h5py.File(test_data_path, 'r') as hf:
    data_test = hf['data_test'][:]


with torch.no_grad():
    G.eval()

    # ________________ Figure 1 ______________________________
    fig1, (ax1,ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20, 4))

    # Get fake data
    num_samples = 2500
    g_input_z = torch.rand(num_samples, noise_dim)
    # g_input_c = torch.rand(num_samples, c_dim)
    if model_type == "d":
        g_input_c = torch.zeros((num_samples, c_dim))
        g_input_c[np.arange(len(g_input_c)), np.random.randint(0, c_dim,
                                                               size=(num_samples,)).tolist()] = 1
    if model_type == "c":
        g_input_c = torch.rand(num_samples, c_dim)
    gen_input = Variable(torch.cat((g_input_z, g_input_c), axis=1)).to(device)
    g_fake_data = G(gen_input).cpu().numpy()

    # Calculate mean at each time step
    time_vec = np.arange(0, data_test.shape[1]) * 15 / 60
    ax1.plot(time_vec, np.mean(data_test, axis=0), label="Real (test set)")
    ax1.plot(time_vec, np.mean(g_fake_data, axis=0), label="Generated")
    # ax1.legend()
    ax1.set_title("Expectation of Load Over Time")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Scaled Power [AC kW]")

    # Calculate std at each time step
    ax2.plot(time_vec, np.std(data_test, axis=0), label="Real (test set)")
    ax2.plot(time_vec, np.std(g_fake_data, axis=0), label="Generated")
    # ax2.legend()
    ax2.set_title("Standard Deviation of Load Over Time")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Scaled Power [AC kW]")

    # Create a ECDF for the real testing data
    data_test_flattened = data_test.flatten()
    x_real = np.sort(data_test_flattened)
    n_real = x_real.shape[0]
    y_real = np.arange(1, n_real + 1) / n_real
    ax3.plot(x_real, y_real, label="Real (test set)")
    ax3.set_ylim([0.7, 1.05])
    ax3.set_xlim([-0.05, 1.5])

    # Create a ECDF for the fake generated data
    g_fake_data_flattened = g_fake_data.flatten()
    x_fake = np.sort(g_fake_data_flattened)
    n_fake = x_fake.shape[0]
    y_fake = np.arange(1, n_fake + 1) / n_fake
    ax3.plot(x_fake, y_fake, label="Generated")
    # ax3.legend()
    ax3.set_title("Empirical CDFs")
    ax3.set_ylabel("Empirical CDF")
    ax3.set_xlabel("Scaled Power [AC kW]")
    ax3.set_ylabel("ECDF")

    # Calculate power spectral density plot
    ax4.psd(data_test_flattened, label="Real (test set)")
    ax4.psd(g_fake_data_flattened, label="Generated")
    # ax4.legend(bbox_to_anchor = (1.5, 0.5))
    ax4.set_title("Power Spectral Density")
    ax4.grid(False)

    fig1.tight_layout()
    plt.savefig('generated_charging_samples/{}_{}_{}_{}.png'.format(training_id, model_number, model_type, "1"), dpi=DPI)

    # ________________ Figure 2 ______________________________
    num_samples = 1000
    g_input_z = torch.rand(num_samples, noise_dim)
    fontsize=11

    if model_type == "c":
        fig2, axs = plt.subplots(8,5, figsize=(15,22))
        for i in range(8): # Different variables
            for j in range(5):   # Different values of variables
                g_input_c = torch.ones((num_samples,c_dim))*0
                c_val = j/4
                g_input_c[:,i] = c_val
                gen_input = Variable(torch.cat((g_input_z, g_input_c), axis=1)).to(device)
                g_fake_data = G(gen_input).cpu().numpy()
                axs[i,j].plot(time_vec, g_fake_data.T)
                # axs[i,j].plot(g_fake_data.T.mean(axis=1))
                axs[i,j].set_ylim([0, 1])
                # axs[i,j].set_title("c{}={}".format(i, c_val))
                axs[i,j].set_title(r"$c_{%d}=%.2f$, fixed $\bf{z}$ & $c_{i,i\neq%d}$" % (i, c_val, i), fontsize=fontsize)
                axs[i,j].set_xlabel("Hour of Day")
                axs[i,j].set_ylabel("Scaled Power [AC kW]")

        fig2.tight_layout()
        plt.savefig('generated_charging_samples/vars_{}_{}_{}.png'.format(training_id, model_number, model_type, "2"), dpi=DPI)

    if model_type == "d":
        fig2, axs = plt.subplots(8, 1, figsize=(12, 22))
        for i in range(8):  # Different variables
            g_input_c = torch.zeros((num_samples, c_dim)) * 0
            g_input_c[:,i] = 1
            gen_input = Variable(torch.cat((g_input_z, g_input_c), axis=1)).to(device)
            g_fake_data = G(gen_input).cpu().numpy()
            axs[i].plot(time_vec, g_fake_data.T)
            # axs[i,j].plot(g_fake_data.T.mean(axis=1))
            axs[i].set_ylim([0, 1])
            # axs[i,j].set_title("c{}={}".format(i, c_val))
            axs[i].set_title(r"$c_{}$".format(i), fontsize=FONT_SIZE)
            axs[i].set_xlabel("Hour of Day")
            axs[i].set_ylabel("Scaled Power [AC kW]")
            plt.figure(figsize=(12, 4))
            plt.plot(time_vec, g_fake_data.T)
            plt.tight_layout()
            plt.ylim([0, 1])
            plt.xlabel("Time [Hour of Day]")
            plt.ylabel("Scaled Power [AC kW]")
            plt.title(r"$c_{}$".format(i),  fontsize=FONT_SIZE)
            plt.savefig('generated_charging_samples/vars_disc_{}_{}_{}_{}.png'.format(training_id, model_number, model_type, i), dpi=DPI)
            np.savetxt('generated_charging_samples/vars_disc_{}_{}_{}_{}.csv'.format(training_id, model_number, model_type, i), g_fake_data, delimiter=",")
            plt.close()
        fig2.tight_layout()
        plt.savefig('generated_charging_samples/vars_disc_{}_{}_{}.png'.format(training_id, model_number, model_type), dpi=DPI)

    # ________________ Figure 3 ______________________________
    if model_type == "c":
        num_samples = 750
        fontsize=14
        g_input_z = torch.rand(num_samples, noise_dim)
        fig3, axs = plt.subplots(8, 4, figsize=(15, 22))
        # G0 through different variables
        for i in range(8):
            # Plot the test set metric for each plot
            axs[i, 0].plot(time_vec, np.mean(data_test, axis=0), "--k", label="Real (test set)")
            axs[i, 1].plot(time_vec, np.std(data_test, axis=0), "--k", label="Real (test set)")

            data_test_flattened = data_test.flatten()
            x_real = np.sort(data_test_flattened)
            n_real = x_real.shape[0]
            y_real = np.arange(1, n_real + 1) / n_real
            axs[i, 2].plot(x_real, y_real, "--k", label="Real (test set)")

            axs[i, 3].psd(data_test_flattened,c='k', ls="--", label="Real (test set)")

            # Different values of each variables
            values = np.array([0, 0.25, 0.5, 0.75, 1, 1.5])
            for j in values:
                g_input_c = torch.ones((num_samples, c_dim)) * 0
                c_val = j
                g_input_c[:, i] = c_val
                gen_input = Variable(torch.cat((g_input_z, g_input_c), axis=1)).to(device)
                g_fake_data = G(gen_input).cpu().numpy()

                # Mean of data
                axs[i,0].plot(time_vec, g_fake_data.T.mean(axis=1), label="$c_{}$={}".format(i, c_val))

                # Std of data
                axs[i,1].plot(time_vec, g_fake_data.T.std(axis=1), label="$c_{}$={}".format(i, c_val))

                # ECDF
                g_fake_data_flattened = g_fake_data.flatten()
                x_fake = np.sort(g_fake_data_flattened)
                n_fake = x_fake.shape[0]
                y_fake = np.arange(1, n_fake + 1) / n_fake
                axs[i,2].plot(x_fake, y_fake, label="$c_{}$={}".format(i, c_val))

                # PSD
                axs[i,3].psd(g_fake_data.flatten(), label="$c_{}$={}".format(i, c_val))

            axs[i, 0].set_ylim([0, 0.8])
            axs[i, 0].set_title(r"Mean Load: Fixed $\bf{z}$ & $c_{i,i\neq%d}$" % i, fontsize=fontsize)
            axs[i, 0].set_xlabel("Hour of Day")
            axs[i, 0].set_ylabel("Scaled Power [AC kW]")

            axs[i, 1].set_ylim([0, 0.8])
            axs[i, 1].set_title(r"Std of Load: Fixed $\bf{z}$ & $c_{i,i\neq%d}$" % i, fontsize=fontsize)
            axs[i, 1].set_xlabel("Hour of Day")
            axs[i, 1].set_ylabel("Scaled Power [AC kW]")

            axs[i, 2].set_title(r"Empirical CDFs: Fixed $\bf{z}$ & $c_{i,i\neq%d}$" % i, fontsize=fontsize)
            axs[i, 2].set_ylim([0.7, 1.05])
            axs[i, 2].set_xlim([-0.05, 1.5])
            axs[i, 2].set_ylabel("ECDF")
            axs[i, 2].set_xlabel("Scaled Power [AC kW]")

            axs[i, 3].set_title(r"Power Spectral Density: Fixed $\bf{z}$ & $c_{i,i\neq%d}$" % i, fontsize=fontsize)
            axs[i, 3].legend(bbox_to_anchor = (1.05, 1))
            axs[i, 3].grid(False)

            fig3.tight_layout()

plt.show()