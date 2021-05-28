import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from ndes_optimizer_rewrited import BasenDESOptimizer as BasenDESOptimizerNew
from ndes_optimizer_original import BasenDESOptimizer as BasenDESOptimizerOld


DEVICE = torch.device("cuda:0")
torch.manual_seed(1)    # reproducible
x_test_cpu = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y_test_cpu = x_test_cpu.pow(2) + 0.2 * torch.rand(x_test_cpu.size())
x_test = x_test_cpu.clone().to(DEVICE)
y_test = y_test_cpu.clone().to(DEVICE)

DRAW_CHARTS = True

x, y = None, None

def get_batches():
    # x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
    global x, y
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(2) + 0.2 * torch.rand(x.size())
    # y = torch.sin(x) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

    x_cuda = x.to(DEVICE)
    y_cuda = y.to(DEVICE)

    torch_dataset = Data.TensorDataset(x_cuda, y_cuda)

    BATCH_SIZE = 64
    # BATCH_SIZE = x_cuda.shape[1]
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE)

    batches = [(idx, x) for idx, x in enumerate(loader)]

    return batches


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


def cycle(batches):
    while True:
        for batch in batches:
            yield batch


def test():
    batches = get_batches()
    # old_score = test_old(batches)
    old_score = 0.0031
    new_score = test_new(batches)

    print(old_score, new_score)

    assert abs(old_score - new_score) <= 0.0005, "The difference between the scores is too high."


def test_old(batches):
    data_gen = cycle(batches)
    model = get_model().to(DEVICE)
    criterion = F.mse_loss

    des_optim = BasenDESOptimizerOld(
        model,
        criterion,
        data_gen,
        restarts=1,
        lower=-2.,
        upper=2.,
        budget=65000,
        tol=1e-6,
        nn_train=True,
        history=16,
        log_best_val=False,
        device=DEVICE,

        x_val=x_test,
        y_val=y_test
    )

    model, fitnesses_iterations = des_optim.run(evaluate_model)

    if DRAW_CHARTS:
        fig, axs = plt.subplots(len(fitnesses_iterations), squeeze=False)
        for idx, fitnesses in enumerate(fitnesses_iterations):
            fitnesses_cpu = [fitness.cpu() for fitness in fitnesses]
            # fitnesses_cpu = fitnesses
            axs[idx, 0].plot(range(len(fitnesses_cpu)), fitnesses_cpu)
        plt.show()

        draw_predictions(model, x_test_cpu, y_test_cpu)

    # print("eval train data: ", evaluate_model_train_data(model))

    return evaluate_model(model)


def test_new(batches):
    model = get_model().to(DEVICE)
    criterion = F.mse_loss
    devices = [torch.device("cuda:0")]

    des_optim = BasenDESOptimizerNew(
        model,
        criterion,
        None,
        restarts=1,
        lower=-2.,
        upper=2.,
        budget=65000,
        tol=1e-6,
        nn_train=True,
        history=16,
        log_best_val=False,
        device=DEVICE,
        devices=devices,
        batches=batches,

        x_val=x_test,
        y_val=y_test
    )

    model, fitnesses_iterations = des_optim.run(evaluate_model)

    if DRAW_CHARTS:
        fig, axs = plt.subplots(len(fitnesses_iterations), squeeze=False)
        # plt.yscale("log")
        for idx, fitnesses in enumerate(fitnesses_iterations):
            fitnesses_cpu = [fitness.cpu() for fitness in fitnesses]
            axs[idx, 0].plot(range(len(fitnesses_cpu)), fitnesses_cpu)
        plt.show()

        draw_predictions(model, x_test_cpu, y_test_cpu)

    # print("eval train data: ", evaluate_model_train_data(model))

    return evaluate_model(model)


def get_model():
    return Net(1, 10, 1)


def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        preds = model(x_test)
    model.train()

    return F.mse_loss(preds, y_test)


def evaluate_model_train_data(model):
    model = model.cpu()

    model.eval()
    with torch.no_grad():
        preds = model(x)
    preds = preds.cpu()
    model.train()

    model.cuda()

    return F.mse_loss(preds, y)


def draw_predictions(model, x_cpu, y_cpu):
    x_cuda = x_cpu.to(DEVICE)

    model.eval()
    with torch.no_grad():
        preds = model(x_cuda).cpu()
        plt.scatter(x_cpu, y_cpu, c='black')
        plt.scatter(x_cpu, preds, c='red')
        plt.show()
    model.train()


if __name__ == "__main__":
    test()
