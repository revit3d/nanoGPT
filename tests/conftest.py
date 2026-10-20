import torch


def pytest_sessionstart(session):
    # Tiny numerical tests are slower with the machine's full CPU thread pool.
    torch.set_num_threads(2)
