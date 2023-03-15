import torch



def test_grid_sample():

    w = 480
    a = torch.arange(1 * 2 * w * w).reshape(1, 2, w, w).float()

    for r in range(w):
        row = r / (w - 1)

        for c in range(w):
            col = c / (w - 1)

            points = torch.tensor([col, row])
            points = points.view(1, 1, 1, 2)
            points = points * 2 - 1

            # set align_cornets to True
            out = torch.nn.functional.grid_sample(a, points, align_corners=True)
            #print(out.tolist(), points.tolist())
            print(out[0, :, 0, 0],  points[0, 0, 0, :])
