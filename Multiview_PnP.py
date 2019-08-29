from utils import compute_transformation, get_camera_intrinsic

def get_fundamental_Matrix():
    # this should be consistent with PR.txt so that the evaluation in 3D wouls be correct.
    F = np.zeros((4, 4), dtype='float64')
    F[0, 0], F[0, 2] = 2356.280953576225, 682.3281173706055
    F[1, 1], F[1, 2] = 2356.280953576225, 522.8933753967285
    F[2, 2] = 1.
    return F

def two_camera_PnP(pr_2D_1, pr_2D_2):
