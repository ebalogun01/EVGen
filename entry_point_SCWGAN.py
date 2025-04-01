import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import torch.utils.data as data_utils
import json
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import pathlib
import logging
import os
import timeit
import datetime
import h5py
import importlib


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def similarity(g_fake_data, c, cs, device):
    # Use this block of code for CONTINUOUS variables
    if c["encoding_type"] == "continuous":
        X = g_fake_data  # Generated vector Shape: (N: inputs count, L: input shape)
        C = cs

        # Get dims of incoming matrices
        N = X.shape[0]  # Batch size
        L = X.shape[1]  # length of the input vector
        m = C.shape[1]  # Number of variables per example

        X_expanded = X.unsqueeze(2).repeat(1, 1, N)  # Repeats the matrix 'X' into the third dimension N times (N, L, N)
        X_subtraction_term = torch.unsqueeze(X, dim=0).transpose(1, 2)  # shape: (1, L, N)
        norm_term = torch.linalg.norm(X_expanded - X_subtraction_term, ord=2, dim=1)  # Take the norm along L's dim
        norm_term = norm_term.unsqueeze(dim=1)  # Shape: (N, 1, N)

        C_expanded = C.unsqueeze(2).repeat(1, 1, N)  # Repeats the matrix 'C' into the third dimension N times (N, m, N)
        C_subtraction_term = torch.unsqueeze(C, dim=0).transpose(1, 2)  # shape: (1, m, N)
        abs_term = torch.abs(C_expanded - C_subtraction_term)  # shape: (N, m, N)

        both_terms = (1 - abs_term) * norm_term + abs_term / (norm_term + 5e-6)  # shape: (N, m, N)
        summed = both_terms.sum(dim=0).sum(dim=1)  # shape: (1, m)
        SC_per_C = summed / (N * (N - 1))  # shape: (1, m)
        SC = torch.mean(SC_per_C)  # scalar

        return SC

    if c["encoding_type"] == "discrete":
        # Use this block of code for DISCRETE variables
        X = g_fake_data
        C = cs
        # Get dims of incoming matrices
        N = X.shape[0]  # Batch size
        L = X.shape[1]  # length of the input vector
        m = C.shape[1]  # Number of variables per example

        X_expanded = X.unsqueeze(2).repeat(1, 1, N)  # Repeats the matrix 'X' into the third dimension N times (N, L, N)
        X_subtraction_term = torch.unsqueeze(X, dim=0).transpose(1, 2)  # shape: (1, L, N)
        norm_term = torch.linalg.norm(X_expanded - X_subtraction_term, ord=2, dim=1)  # Take the norm along L's dim
        norm_term = norm_term.unsqueeze(dim=1)  # Shape: (N, 1, N)
        # assert norm_term.shape == (N, 1, N)

        C_expanded = C.unsqueeze(2).repeat(1, 1, N)  # Repeats the matrix 'C' into the third dimension N times (N, m, N)
        C_dot_term = torch.unsqueeze(C, dim=0).transpose(1, 2)  # shape: (1, m, N)

        # Now take the dot product...
        dot_term = (C_expanded * C_dot_term).sum(dim=1) # summed along the 'm' dimension for dot product
        dot_term = dot_term.unsqueeze(dim=1) #  to bring back shape (N, 1, N)
        # assert dot_term.shape == (N, 1, N)
        both_terms = dot_term * norm_term + (1 - dot_term)/(norm_term + 5e-6)
        summed = both_terms.sum()
        SC = summed / (N * (N - 1))

        return SC




def gradient_penalty(real, fake, c, device):
    lam = c["gp_reg"]
    eta = torch.rand(c["samples_per_batch"], 1, device=device)
    eta = eta.expand_as(real)

    interpolated = eta * real + ((1 - eta) * fake)
    interpolated = Variable(interpolated, requires_grad=True)

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                              grad_outputs=torch.ones(prob_interpolated.size(), device=device),
                              create_graph=True, retain_graph=True)[0]
    grad_norm = gradients.norm(2, dim=1)
    grad_penalty = ((grad_norm - 1) ** 2).mean() * lam

    return grad_penalty


def sine_waves(c):
    num_samples = 18944
    amplitude = np.random.uniform(low=3, high=8, size=(num_samples, 1))
    freq = np.random.uniform(low=3, high=8, size=(num_samples, 1))
    theta = 0
    start_time = 0
    time = np.arange(start_time, c["fake_training"]["end_time"], 1 / (c["fake_training"]["sample_rate"]))
    sinewave = amplitude * np.sin(2 * np.pi * freq * time + theta)

    return sinewave


def train(c):
    # Create results directories
    photo_path = os.path.join(c["results_dir"], "plot_images")
    pathlib.Path(c["results_dir"]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(c["results_dir"], "models").mkdir(parents=True, exist_ok=True)
    pathlib.Path(photo_path).mkdir(parents=True, exist_ok=True)

    # Make Logging file
    PID = os.getpid()
    logging_path = os.path.join(c["results_dir"], "output.out")
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s', '%m/%d/%Y %I:%M:%S')
    hdlr1 = logging.FileHandler(logging_path)
    hdlr1.setFormatter(formatter)
    logger = logging.getLogger(str(PID))
    logger.handlers = []
    logger.addHandler(hdlr1)
    logger.setLevel(logging.INFO)

    logging.info("Using WGAN architecture")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Training on {}".format(device))

    # Set default floating point precision
    if torch.cuda.is_available():
        default_dtype = torch.cuda.FloatTensor
        torch.set_default_tensor_type(default_dtype)
    else:
        default_dtype = torch.FloatTensor
        torch.set_default_tensor_type(default_dtype)

    # Set random seeds
    if isinstance(c["torch_seed"], int):
        torch.manual_seed(c["torch_seed"])
    if isinstance(c["np_seed"], int):
        np.random.seed(c["torch_seed"])

    # Read in data (each row is a separate sample). Figure out time dimension
    if c["fake_training"]["use"]:
        data = sine_waves(c)
        c["seq_length"] = data.shape[1]
    else:
        data_list = list()
        for file in c["data_file"]:
            data = np.load(file)
            data_list.append(data)
        data = np.vstack(data_list)
        c["seq_length"] = data.shape[1]

    # Split the data into train and test sets
    mask = np.zeros(data.shape[0])
    indices = np.random.choice(np.arange(data.shape[0]), replace=False, size=int(c["num_test_samples"]))
    mask[indices] = 1
    data_train = data[mask == 0]
    data_test = data[mask == 1]
    logger.info("Train/Test Split: {}% / {}%".format(round(data_train.shape[0] * 100 / data.shape[0], 1),
                                                     round(data_test.shape[0] * 100 / data.shape[0], 1)))

    # paper_imgs statistics
    num_samples = data_train.shape[0]
    batches_per_epoch = num_samples / c["samples_per_batch"]
    total_iterations = batches_per_epoch * c["num_epochs"]

    # Normalize data
    data_stats = {"max": data_train.max(), "min": data_train.min()}
    c["data_stats"] = data_stats
    data_train = (data_train - data_stats["min"]) / (data_stats["max"] - data_stats["min"])
    data_test = (data_test - data_stats["min"]) / (data_stats["max"] - data_stats["min"])
    print('ok')
    # Save data to disk for later use (right now, it is being saved AFTER normalization)
    with h5py.File(os.path.join(c["results_dir"], "data_test.h5"), 'w') as hf:
        hf.create_dataset("data_test", data=data_test)
    with h5py.File(os.path.join(c["results_dir"], "data_train.h5"), 'w') as hf:
        hf.create_dataset("data_train", data=data_train)

    # Send data to tensors (if the entire dataset can fit in RAM, send it to device in the next line)
    train_tensor = torch.from_numpy(data_train).type(default_dtype).to(device)
    train = data_utils.TensorDataset(train_tensor)
    train_loader = data_utils.DataLoader(train, batch_size=c["samples_per_batch"], shuffle=True)
    # train_loader = FastTensorDataLoader(train_tensor, batch_size=c["samples_per_batch"], shuffle=True)

    # Set up pytorch models
    global D  # Discriminator
    global G  # Generator
    gan = importlib.import_module(c["network_module"])
    if c["resume"]["status"]:
        models = torch.load(os.path.join(c["results_dir"], "models", "model_{}".format(c["resume"]["resume_from"])))
        D = models["D_model"]
        G = models["G_model"]
        epoch_start = models["epoch_num"]
        total_batch_tally = models["batch_num"]
        logger.info("Resuming training for {}".format(c["results_dir"]))
    else:
        D = gan.Discriminator(c)
        G = gan.Generator(c)
        epoch_start = 1
        total_batch_tally = 1
        logger.info("Starting new training for {}".format(c["results_dir"]))

    D.to(device)
    G.to(device)

    # Set optimizers
    d_optimizer = optim.Adam(D.parameters(), lr=c["d_learning_rate"], betas=(c["adam_beta1"], c["adam_beta2"]),
                             weight_decay=c["weight_decay"], amsgrad=False)
    g_optimizer = optim.Adam(G.parameters(), lr=c["g_learning_rate"], betas=(c["adam_beta1"], c["adam_beta2"]),
                             weight_decay=c["weight_decay"], amsgrad=False)

    # Make tensorboard object and write configs file to results directory. This will make the file if it doesnt exist already
    writer = SummaryWriter(c["results_dir"])
    path = os.path.join(c["results_dir"], "configs.json")
    with open(path, 'w') as fp:
        json.dump(c, fp, indent=1)

    # Run through epochs
    train_start_time = timeit.default_timer()
    remaining_time = "inf"
    for epoch_id in range(epoch_start, c["num_epochs"] + 1):

        # Adjust the learning rate, if lr scheduling is turned on
        if c["lr_schedule"]["use"] and epoch_id % c["lr_schedule"]["freq"] == 0:
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * c['lr_schedule']['factor']
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * c['lr_schedule']['factor']
            logger.info("Changing learning rate")

        # Run through mini-batches
        for batch_id, train_data in enumerate(train_loader, start=1):
            D.train()
            G.train()
            # d_real_data = train_data[0].to(device)
            d_real_data = train_data[0]

            # Run each batch through each network a certain number of times, separately
            # Train the Discriminator
            d_start_time = timeit.default_timer()
            for d_index in range(c["d_steps"]):
                D.zero_grad()

                #  Part 1: Train D on real
                d_real_decision = D.forward(d_real_data)  # changed to D.forward
                d_real_error = torch.mean(d_real_decision)

                #  Part 2: Train D on fake
                d_input_z = torch.rand(c["samples_per_batch"], c["g_input_size"], device=device, requires_grad=False)
                if c["encoding_type"] == "discrete":
                    d_input_c = torch.zeros((c["samples_per_batch"], c["c_dim"]), device=device, requires_grad=False)
                    d_input_c[np.arange(len(d_input_c)), np.random.randint(0, c["c_dim"],
                                                                           size=(c["samples_per_batch"],)).tolist()] = 1
                if c["encoding_type"] == "continuous":
                    d_input_c = torch.rand(c["samples_per_batch"], c["c_dim"], device=device, requires_grad=False)
                d_gen_input = Variable(torch.cat((d_input_z, d_input_c), axis=1))
                d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
                d_fake_decision = D(d_fake_data)
                d_fake_error = torch.mean(d_fake_decision)

                # Part 3: Calculate Gradient Penalty
                grad_penalty = gradient_penalty(d_real_data, d_fake_data, c, device)
                # grad_penalty = 0

                # Part 4: Backward prop and gradient step
                total_d_error_without_gp = -1 * (d_real_error - d_fake_error)
                total_d_error = total_d_error_without_gp + grad_penalty
                # total_d_error = -1 * (d_real_error - d_fake_error)
                total_d_error.backward()
                d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
            d_time = (timeit.default_timer() - d_start_time) * 1000 / c["d_steps"]

            # Train the Generator
            g_start_time = timeit.default_timer()
            for g_index in range(c["g_steps"]):
                # Train G on D's response (but DO NOT train D on these labels) (input to G is torch.Size([500, 1]))
                G.zero_grad()

                # Train G to pretend it's genuine
                g_input_z = torch.rand(c["samples_per_batch"], c["g_input_size"], device=device)
                if c["encoding_type"] == "discrete":
                    g_input_c = torch.zeros((c["samples_per_batch"], c["c_dim"]), device=device)
                    g_input_c[np.arange(len(g_input_c)), np.random.randint(0, c["c_dim"],
                                                                           size=(c["samples_per_batch"],)).tolist()] = 1
                if c["encoding_type"] == "continuous":
                    g_input_c = torch.rand(c["samples_per_batch"], c["c_dim"], device=device)
                gen_input = Variable(torch.cat((g_input_z, g_input_c), axis=1))
                g_fake_data = G(gen_input)
                dg_fake_decision = D(g_fake_data)
                g_error_term1 = -torch.mean(dg_fake_decision)

                # Calculate a similarity constraint
                sim = similarity(g_fake_data, c, g_input_c, device)
                sim_error = c["sc_reg"] * sim
                g_error = g_error_term1 + sim_error

                g_error.backward()
                g_optimizer.step()  # Only optimizes G's parameters
            g_time = (timeit.default_timer() - g_start_time) * 1000 / c["g_steps"]

            # Write data to tensorboard
            if total_batch_tally % c["tensorboard_interval"] == 0:
                writer.add_scalar("Discriminator/Discriminator Loss: D(x) - D(G(z))",
                                  -1 * total_d_error_without_gp.data.item(),
                                  total_batch_tally)
                writer.add_scalar("Discriminator/Gradient Penalty", grad_penalty.data.item(), total_batch_tally)
                writer.add_scalar("Discriminator/Minimizing Function", total_d_error.data.item(), total_batch_tally)
                writer.add_scalar("Generator/Generator Loss: D(G(z))", -1 * g_error_term1.data.item(),
                                  total_batch_tally)
                writer.add_scalar("Generator/Similarity Constraint", sim_error.data.item(), total_batch_tally)
                writer.add_scalar("Generator/Minimizing Function", g_error.data.item(), total_batch_tally)
                writer.add_scalars("Computation Times", {"Discriminator Step": d_time,
                                                         "Generator Step": g_time}, total_batch_tally)

            # Write a plot of a fake generated output to tensorboard every once in a while
            if total_batch_tally % c["plot_interval"] == 0:
                with torch.no_grad():
                    G.eval()
                    fig1, ax1 = plt.subplots()
                    g_input_z = torch.rand(c["samples_per_batch"], c["g_input_size"], device=device)
                    if c["encoding_type"] == "discrete":
                        g_input_c = torch.zeros((c["samples_per_batch"], c["c_dim"]), device=device)
                        g_input_c[np.arange(len(g_input_c)), np.random.randint(0, c["c_dim"],
                                                                               size=(
                                                                               c["samples_per_batch"],)).tolist()] = 1
                    if c["encoding_type"] == "continuous":
                        g_input_c = torch.rand(c["samples_per_batch"], c["c_dim"], device=device)
                    gen_input = Variable(torch.cat((g_input_z, g_input_c), axis=1))
                    g_fake_data = G(gen_input)
                    ax1.plot(np.array(g_fake_data.detach().tolist()).T[:, 0:5])
                    plt.text(0, 1, 'Batch #: {}'.format(total_batch_tally), horizontalalignment='left',
                             verticalalignment='bottom', transform=ax1.transAxes)
                    ax1.set_title("Fake Generated Output")
                    ax1.set_ylim([0, 1])
                    fig1.tight_layout()
                    if c["save_plots"]:
                        plt.savefig(os.path.join(photo_path, "{}.png".format(total_batch_tally)), dpi=70)
                    plt.clf()
                    plt.close('all')

            if total_batch_tally % c["save_interval"] == 0:
                model_dict = {'epoch_num': epoch_id, 'batch_num': total_batch_tally, 'D_model': D, 'G_model': G}
                torch.save(model_dict, os.path.join(c["results_dir"], "models", "model_{}".format(total_batch_tally)))

            if total_batch_tally % c["print_interval"] == 0:
                logger.info(
                    "Completed: Epoch {}/{} \t Batch {}/{} \t Total Batches {}/{} \t Time left {}".format(epoch_id, c[
                        "num_epochs"], batch_id, batches_per_epoch, total_batch_tally, total_iterations,
                                                                                                          remaining_time))
            total_batch_tally = total_batch_tally + 1

        total_time = timeit.default_timer() - train_start_time
        time_per_epoch = total_time / epoch_id
        remaining_time = str(datetime.timedelta(seconds=(c["num_epochs"] - epoch_id) * time_per_epoch))

    # When training is done, wrap up the tensorboard files
    writer.flush()
    writer.close()


if __name__ == "__main__":
    with open("configs.json", "r") as read_file:
        config = json.load(read_file)
    train(config)
