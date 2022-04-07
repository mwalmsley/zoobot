import torch


def dummy_dirichlet(votes, total_counts):
    assert all(total_counts > 0)
    return torch.rand_like(total_counts)


if __name__ == '__main__':


    votes = torch.stack([torch.arange(0, 3) for n in range(5)])
    votes[2] = torch.tensor(data=[0, 0, 0])
    # print(votes)


    total_counts = torch.sum(votes, dim=1, dtype=torch.float32)
    # print(total_counts)

    # print(total_counts == 0)

    indices = torch.arange(0, len(total_counts), dtype=torch.long)
    # print(indices)

    nonzero_indices = indices[total_counts != 0]
    print(nonzero_indices)

    nonzero_votes = votes[nonzero_indices]
    print(nonzero_votes)
    nonzero_total_counts = total_counts[nonzero_indices]
    print(nonzero_total_counts)

    nonzero_log_probs = dummy_dirichlet(nonzero_votes, nonzero_total_counts)

    # mix back together
    log_probs = torch.zeros_like(total_counts).scatter_(dim=0, index=nonzero_indices, src=nonzero_log_probs)
    print(log_probs)
    