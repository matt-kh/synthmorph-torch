import cv2 as cv
import numpy as np
import math

random_state = np.random.RandomState(None)


def draw_polygon(size, max_sides=8):
    """ Draw a polygon with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
    """
    img = np.zeros(shape=size, dtype=np.float32)
    num_corners = random_state.randint(3, max_sides)
    min_dim = min(img.shape[0], img.shape[1])
    rad = max(random_state.rand() * min_dim / 2, min_dim / 10)
    x = random_state.randint(rad, img.shape[1] - rad)  # Center of a circle
    y = random_state.randint(rad, img.shape[0] - rad)

    # Sample num_corners points inside the circle
    slices = np.linspace(0, 2 * math.pi, num_corners + 1)
    angles = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
              for i in range(num_corners)]
    points = np.array([[int(x + max(random_state.rand(), 0.4) * rad * math.cos(a)),
                        int(y + max(random_state.rand(), 0.4) * rad * math.sin(a))]
                       for a in angles])

    # Filter the points that are too close or that have an angle too flat
    norms = [np.linalg.norm(points[(i-1) % num_corners, :]
                            - points[i, :]) for i in range(num_corners)]
    mask = np.array(norms) > 0.01
    points = points[mask, :]
    num_corners = points.shape[0]
    corner_angles = [angle_between_vectors(points[(i-1) % num_corners, :] -
                                           points[i, :],
                                           points[(i+1) % num_corners, :] -
                                           points[i, :])
                     for i in range(num_corners)]
    mask = np.array(corner_angles) < (2 * math.pi / 3)
    points = points[mask, :]
    num_corners = points.shape[0]
    if num_corners < 3:  # not enough corners
        return draw_polygon(img, max_sides)

    corners = points.reshape((-1, 1, 2))
    col = 1
    cv.fillPoly(img, [corners], col)
    return img, points


def draw_multiple_polygons(size, nb_polygons, max_sides=8):
    """ Draw multiple polygons with a random number of corners
    and return the corner points
    Parameters:
      max_sides: maximal number of sides + 1
      nb_polygons: maximal number of polygons
    """
    img = np.zeros(shape=size, dtype=np.float32)
    segments = np.empty((0, 4), dtype=np.int32)
    centers = []
    rads = []
    points = np.empty((0, 2), dtype=np.int32)
    label = 1
    for i in range(nb_polygons):
        num_corners = random_state.randint(3, max_sides)
        num_corners = max_sides - 1
        min_dim = min(img.shape[0], img.shape[1])
        rad = max(random_state.rand() * min_dim / 2, min_dim / 10)
        x = random_state.randint(rad, img.shape[1] - rad)  # Center of a circle
        y = random_state.randint(rad, img.shape[0] - rad)

        # Sample num_corners points inside the circle
        slices = np.linspace(0, 2 * math.pi, num_corners + 1)
        angles = [slices[i] + random_state.rand() * (slices[i+1] - slices[i])
                  for i in range(num_corners)]
        new_points = [[int(x + max(random_state.rand(), 0.4) * rad * math.cos(a)),
                       int(y + max(random_state.rand(), 0.4) * rad * math.sin(a))]
                      for a in angles]
        new_points = np.array(new_points)

        # Filter the points that are too close or that have an angle too flat
        norms = [np.linalg.norm(new_points[(i-1) % num_corners, :]
                                - new_points[i, :]) for i in range(num_corners)]
        mask = np.array(norms) > 0.01
        new_points = new_points[mask, :]
        num_corners = new_points.shape[0]
        corner_angles = [angle_between_vectors(new_points[(i-1) % num_corners, :] -
                                               new_points[i, :],
                                               new_points[(i+1) % num_corners, :] -
                                               new_points[i, :])
                         for i in range(num_corners)]
        mask = np.array(corner_angles) < (2 * math.pi / 3)
        new_points = new_points[mask, :]
        num_corners = new_points.shape[0]
        if num_corners < 3:  # not enough corners
            continue

        new_segments = np.zeros((1, 4, num_corners))
        new_segments[:, 0, :] = [new_points[i][0] for i in range(num_corners)]
        new_segments[:, 1, :] = [new_points[i][1] for i in range(num_corners)]
        new_segments[:, 2, :] = [new_points[(i+1) % num_corners][0]
                                 for i in range(num_corners)]
        new_segments[:, 3, :] = [new_points[(i+1) % num_corners][1]
                                 for i in range(num_corners)]

        # Check that the polygon will not overlap with pre-existing shapes
        if intersect(segments[:, 0:2, None],
                     segments[:, 2:4, None],
                     new_segments[:, 0:2, :],
                     new_segments[:, 2:4, :],
                     3) or overlap(np.array([x, y]), rad, centers, rads):
            continue
        centers.append(np.array([x, y]))
        rads.append(rad)
        new_segments = np.reshape(np.swapaxes(new_segments, 0, 2), (-1, 4))
        segments = np.concatenate([segments, new_segments], axis=0)

        # Color the polygon with a custom background
        corners = new_points.reshape((-1, 1, 2))
        mask = np.zeros(img.shape, np.uint8)
        background = np.ones(shape=size, dtype=np.float32) * label
        cv.fillPoly(mask, [corners], 255)
        locs = np.where(mask != 0)
        img[locs[0], locs[1]] = background[locs[0], locs[1]]
        points = np.concatenate([points, new_points], axis=0)
        label += 1

    return img, points


def draw_ellipses(img, nb_ellipses=20):
    """ Draw several ellipses
    Parameters:
      nb_ellipses: maximal number of ellipses
    """
    centers = np.empty((0, 2), dtype=np.int)
    rads = np.empty((0, 1), dtype=np.int)
    min_dim = min(img.shape[0], img.shape[1]) / 4
    background_color = int(np.mean(img))
    for i in range(nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        col = get_random_color(background_color)
        angle = random_state.rand() * 90
        cv.ellipse(img, (x, y), (ax, ay), angle, 0, 360, col, -1)
    return np.empty((0, 2), dtype=np.int)


def ccw(A, B, C, dim):
    """ Check if the points are listed in counter-clockwise order """
    if dim == 2:  # only 2 dimensions
        return((C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0])
               > (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0]))
    else:  # dim should be equal to 3
        return((C[:, 1, :] - A[:, 1, :])
               * (B[:, 0, :] - A[:, 0, :])
               > (B[:, 1, :] - A[:, 1, :])
               * (C[:, 0, :] - A[:, 0, :]))
    

def intersect(A, B, C, D, dim):
    """ Return true if line segments AB and CD intersect """
    return np.any((ccw(A, C, D, dim) != ccw(B, C, D, dim)) &
                  (ccw(A, B, C, dim) != ccw(A, B, D, dim)))


def overlap(center, rad, centers, rads):
    """ Check that the circle with (center, rad)
    doesn't overlap with the other circles """
    flag = False
    for i in range(len(rads)):
        if np.linalg.norm(center - centers[i]) + min(rad, rads[i]) < max(rad, rads[i]):
            flag = True
            break
    return flag


def keep_points_inside(points, size):
    """ Keep only the points whose coordinates are inside the dimensions of
    the image of size 'size' """
    mask = (points[:, 0] >= 0) & (points[:, 0] < size[1]) &\
           (points[:, 1] >= 0) & (points[:, 1] < size[0])
    return points[mask, :]


def angle_between_vectors(v1, v2):
    """ Compute the angle (in rad) between the two vectors v1 and v2. """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_random_color(background_color):
    """ Output a random scalar in grayscale with a least a small
        contrast with the background color """
    color = random_state.randint(256)
    if abs(color - background_color) < 30:  # not enough contrast
        color = (color + 128) % 256
    return color


def generate_background(size=(960, 1280), nb_blobs=100, min_rad_ratio=0.01,
                        max_rad_ratio=0.05, min_kernel_size=50, max_kernel_size=300):
    """ Generate a customized background image
    Parameters:
      size: size of the image
      nb_blobs: number of circles to draw
      min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
      max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
      min_kernel_size: minimal size of the kernel
      max_kernel_size: maximal size of the kernel
    """
    img = np.zeros(size, dtype=np.uint8)
    dim = max(size)
    cv.randu(img, 0, 255)
    cv.threshold(img, random_state.randint(256), 255, cv.THRESH_BINARY, img)
    background_color = int(np.mean(img))
    blobs = np.concatenate([random_state.randint(0, size[1], size=(nb_blobs, 1)),
                            random_state.randint(0, size[0], size=(nb_blobs, 1))],
                           axis=1)
    for i in range(nb_blobs):
        col = get_random_color(background_color)
        cv.circle(img, (blobs[i][0], blobs[i][1]),
                  np.random.randint(int(dim * min_rad_ratio),
                                    int(dim * max_rad_ratio)),
                  col, -1)
    kernel_size = random_state.randint(min_kernel_size, max_kernel_size)
    cv.blur(img, (kernel_size, kernel_size), img)
    return img