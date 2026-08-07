import torch


def compute_overlaps(metadatas):
    """Return the last contiguous future volume intersecting each volume."""
    N = len(metadatas)
    maxidx = []

    for i in range(N):
        curr_metadata = metadatas[i]
        curr_maxidx = i
        for ii in range(i, N):
            if curr_metadata.intersects(metadatas[ii]):
                curr_maxidx = ii
            else:
                break
        maxidx.append(curr_maxidx)

    return torch.tensor(maxidx)
