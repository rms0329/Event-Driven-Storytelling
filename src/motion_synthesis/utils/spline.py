import matplotlib.pyplot as plt
import numpy as np


def compute_tcb_spline(control_points, tension=0, continuity=0, bias=0, num_points=100):
    """
    Compute the TCB spline for a given set of control points in 3D.
    points: Control points array in 3D (B, N, 3)
    tension: Tension parameter
    continuity: Continuity parameter
    bias: Bias parameter
    num_points: Number of points to sample between each pair of control points
    """
    B, N, _ = control_points.shape
    ks = np.tile(np.arange(N - 1)[:, None], (1, num_points))  # (N-1, num_points)
    ts = np.tile(np.linspace(0, 1, num_points)[None, :], (N - 1, 1))  # (N-1, num_points)
    assert ks.shape == ts.shape

    spline_points = interpolate_tcb_spline(
        control_points,
        ks.flatten(),
        ts.flatten(),
        tension,
        continuity,
        bias,
    )  # (B, (N-1)*num_points, 3)
    return spline_points


def interpolate_tcb_spline(control_points, ks, ts, tension=0, continuity=0, bias=0):
    """
    control_points: Control points array in 3D (B, N, 3)
    ks: Indices of the start control points (L,)
    ts: Interpolation parameters (L,)
    """
    B, N, _ = control_points.shape
    incoming, outgoing = calculate_tangents(control_points, tension, continuity, bias)  # (B, N, 3)

    p0 = control_points[:, ks, :]  # (B, L, 3)
    p1 = control_points[:, ks + 1, :]  # (B, L, 3)
    m0 = outgoing[:, ks, :]  # (B, L, 3)
    m1 = incoming[:, ks + 1, :]  # (B, L, 3)
    ts = np.tile(ts[None, :], (B, 1))  # (B, L)

    points = cubic_hermite_spline(
        p0.reshape(-1, 3),
        p1.reshape(-1, 3),
        m0.reshape(-1, 3),
        m1.reshape(-1, 3),
        ts.reshape(-1, 1),
    )
    return points.reshape(B, -1, 3)  # (B, L, 3)


def calculate_tangents(control_points, tension=0, continuity=0, bias=0):
    """
    Calculate tangents for the control points based on tension and continuity in 3D.
    points: Control points array in 3D (B, N, 3)
    tension: Tension parameter
    continuity: Continuity parameter
    bias: Bias parameter
    """

    def _reshape_parameters(tension, continuity, bias):
        ret = []
        for param in [tension, continuity, bias]:
            if isinstance(param, (int, float)):  # when a single value is given
                param = np.array([param]).reshape(1, 1, 1)
            if param.ndim == 1:  # when param is given for each control point
                param = param[None, :, None]  # (1, N, 1)
            elif param.ndim == 2:
                param = param[:, :, None]  # (B, N, 1)
            ret.append(param)
        return ret

    tension, continuity, bias = _reshape_parameters(tension, continuity, bias)
    p0, p2 = np.zeros_like(control_points), np.zeros_like(control_points)
    p0[:, 1:, :] = control_points[:, :-1, :]  # i-1th control point
    p2[:, :-1, :] = control_points[:, 1:, :]  # i+1th control point
    p1 = control_points

    # this values must not be used
    p0[:, 0, :] = np.nan
    p2[:, -1, :] = np.nan

    # compute the tangents for the intermediate control points
    incoming = 0.5 * (1 - tension) * (1 - continuity) * (1 + bias) * (p1 - p0) + 0.5 * (1 - tension) * (1 + continuity) * (1 - bias) * (p2 - p1)  # fmt: skip
    outgoing = 0.5 * (1 - tension) * (1 + continuity) * (1 + bias) * (p1 - p0) + 0.5 * (1 - tension) * (1 - continuity) * (1 - bias) * (p2 - p1)  # fmt: skip

    # boundary conditions
    outgoing[:, 0, :] = 0.5 * (1 - tension[:, 0, :]) * (1 - continuity[:, 0, :]) * (1 - bias[:, 0, :]) * (p2[:, 0, :] - p1[:, 0, :])  # fmt: skip
    incoming[:, 0, :] = outgoing[:, 0, :]
    incoming[:, -1, :] = 0.5 * (1 - tension[:, -1, :]) * (1 - continuity[:, -1, :]) * (1 + bias[:, -1, :]) * (p1[:, -1, :] - p0[:, -1, :])  # fmt: skip
    outgoing[:, -1, :] = incoming[:, -1, :]

    return incoming, outgoing


def cubic_hermite_spline(p0, p1, m0, m1, t):
    """
    Compute a point on a cubic Hermite spline segment in 3D.
    p0: Start point (B, 3)
    p1: End point (B, 3)
    m0: Start tangent (B, 3)
    m1: End tangent (B, 3)
    t: Interpolation parameter (B,) or (B, 1)
    """
    if isinstance(t, (int, float)):
        t = np.array([t]).reshape(1, 1)
    if t.ndim == 1:  # (B,)
        t = t[:, None]  # (B, 1)

    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1  # (B, 3)


if __name__ == "__main__":
    control_points = np.array(
        [
            [
                [0, 0, 0],
                [1, 2, 1],
                [2, 0, 2],
                [3, 3, 3],
            ],
            [
                [0, 0, 0],
                [1, 2, 0],
                [3, 0, 0],
                [1, 1, 3],
            ],
        ],
        dtype=np.float32,
    )
    tension = np.array([0, 0, 0, 0])  # Example tension values
    continuity = np.array([0, 0, 0, 0])  # Example continuity values
    bias = np.array([0, 0, 0, 0])  # Example bias values
    spline_points = compute_tcb_spline(control_points, tension, continuity, bias, num_points=50)

    batch_size = control_points.shape[0]
    for b in range(batch_size):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot spline
        ax.plot(
            spline_points[b, :, 0],
            spline_points[b, :, 1],
            spline_points[b, :, 2],
            "r-",
            label="Spline",
        )

        # Plot control points
        ax.plot(
            control_points[b, :, 0],
            control_points[b, :, 1],
            control_points[b, :, 2],
            "bo-",
            label="Control Points",
        )

        plt.title(f"3D TCB Spline Visualization (Batch {b})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.legend()
        plt.show()
