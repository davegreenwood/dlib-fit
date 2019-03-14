"""Fit dlib 68 to images or video."""
import logging
import os
import json
import numpy as np
import dlib
import imageio
import click
from pkg_resources import resource_filename


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

MFNAME = "model/shape_predictor_68_face_landmarks.dat"
MODEL = resource_filename(__name__, MFNAME)
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(MODEL)
LOG = logging.getLogger(__name__)

# reverse the model idxs
FLIP_IDX = list(range(17))[::-1] + \
    [26, 25, 24, 23, 22] + \
    [21, 20, 19, 18, 17] + \
    [27, 28, 29, 30] + \
    [35, 34, 33, 32, 31] + \
    [45, 44, 43, 42, 47, 46] + \
    [39, 38, 37, 36, 41, 40] + \
    [54, 53, 52, 51, 50, 49, 48] + \
    [59, 58, 57, 56, 55] + \
    [64, 63, 62, 61, 60] + \
    [67, 66, 65]


# -----------------------------------------------------------------------------
# landmarks
# -----------------------------------------------------------------------------


def set_logger(verbose, logger):
    """Set the logger level and format."""
    levels = {0: logging.CRITICAL, 1: logging.WARN,
              2: logging.INFO, 3: logging.DEBUG}
    level = levels.get(verbose, logging.DEBUG)
    fstr = "%(asctime)s : %(name)s : %(levelname)s : %(message)s"
    logging.basicConfig(level=level, format=fstr)
    logger.info("Logger set, level: %d", level)


def shape_to_np(shape, dtype="int"):
    """Convert the dlib landmarks to a numpy array."""
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rgb_to_grey(image):
    """Convert to greyscale."""
    grey = image[:, :, :3] @ [0.299, 0.587, 0.114]
    return grey.astype(np.uint8)


def flip_lmk(image):
    """The mirror of the landmarks."""
    gray = rgb_to_grey(image)[:, ::-1]
    y, x = gray.shape[:2]
    shape = get_lmk(gray)
    # flip values
    shape[:, 0] = x - shape[:, 0]
    # flip idxs
    return shape[FLIP_IDX, :]


def forward_lmk(image):
    """The landmarks of the face in the image"""
    gray = rgb_to_grey(image)
    return get_lmk(gray)


def get_lmk(gray):
    """Detect a face and 68 landmarks in a gray image """
    rect = DETECTOR(gray, 0)[0]
    return shape_to_np(PREDICTOR(gray, rect))


def landmarks(image):
    """Return the mean of forward and flipped landmarks """
    lmk_f = forward_lmk(image) * 0.5
    lmk_r = flip_lmk(image) * 0.5
    return (lmk_f + lmk_r).astype(np.int32)

# -----------------------------------------------------------------------------
# data io
# -----------------------------------------------------------------------------


def save_json(lmks, save_dir):
    """Save the lmks dictionary."""
    base_name = lmks["fname"].split(".")[:-1]
    frame = "_{:06d}".format(int(lmks["frame"]))
    jfname = "".join(base_name + [frame, ".json"])
    with open(os.path.join(save_dir, jfname), "w") as fid:
        json.dump(lmks, fid)


def image_fname(vfname, frame, save_dir, extn=".png"):
    """Form the image name from the video."""
    base_name = os.path.split(vfname)[-1].split(".")[:-1]
    frame_num = "_{:06d}".format(frame)
    return os.path.join(save_dir, "".join(base_name + [frame_num, extn]))


def extract_images(vfname, start, end, save_root):
    for frame, image in yield_idx_image(vfname):
        if frame < start:
            continue
        if frame == end:
            return
        fname = image_fname(vfname, frame, save_root)
        imageio.imwrite(fname, image)


def yield_idx_image(vfname):
    """Yield image and frame index as tuple, from video file.
    Yield:
        (frame, image) {tuple} -- yield from each iteration.
    Arguments:
        vfname {str} -- filename of video
    """
    reader = imageio.get_reader(vfname)
    frame = -1
    try:
        for frame, image in enumerate(reader):
            yield frame, image
    except imageio.core.format.CannotReadFrameError as _:
        msg = "Could not read frame {} - aborting.".format(frame + 1)
        LOG.warning(msg)
        return


def yield_landmarks(vfname, start=0, end=-1):
    """ Yield detected poses from a video file.
    Optionally start frames at the start."""
    _, fname = os.path.split(vfname)
    for frame, image in yield_idx_image(vfname):
        if frame < start:
            continue
        if frame == end:
            return
        try:
            lmks = landmarks(image)
        except Exception as e:
            lmks = []
            LOG.exception(e)
        yield dict(fname=fname, frame=frame, landmarks_2d=lmks.tolist())


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@click.command()
@click.argument("vfname", type=click.Path(exists=True, dir_okay=False))
@click.argument("save_root", type=click.Path(exists=True, file_okay=False))
@click.option("--verbose", "-v", type=click.INT, default=2,
              help="verbosity level")
@click.option("--start", "-k", type=click.INT, default=0,
              help="start at K frames")
@click.option("--end", "-n", type=click.INT, default=-1,
              help="end at K frames")
def track(vfname, save_root, **kwargs):
    """Main function call"""
    k = kwargs.get("start", 0)
    n = kwargs.get("end", -1)
    v = kwargs.get("verbose", 2)
    set_logger(v, LOG)
    LOG.info("Reading: %s", vfname)
    LOG.info("Saving to: %s", save_root)
    LOG.info("Skipping first %d frames of video.", k)
    LOG.info("Stopping after %d frames of video.", n-1)
    for lmk in yield_landmarks(vfname, k, n):
        save_json(lmk, save_root)


@click.command()
@click.argument("vfname", type=click.Path(exists=True, dir_okay=False))
@click.argument("save_root", type=click.Path(exists=True, file_okay=False))
@click.option("--verbose", "-v", type=click.INT, default=2,
              help="verbosity level")
@click.option("--start", "-k", type=click.INT, default=0,
              help="start at K frames")
@click.option("--end", "-n", type=click.INT, default=-1,
              help="end at K frames")
def extract(vfname, save_root, **kwargs):
    """Main function call"""
    k = kwargs.get("start", 0)
    n = kwargs.get("end", -1)
    v = kwargs.get("verbose", 2)
    set_logger(v, LOG)
    LOG.info("Reading: %s", vfname)
    LOG.info("Saving to: %s", save_root)
    LOG.info("Skipping first %d frames of video.", k)
    LOG.info("Stopping after %d frames of video.", n-1)
    extract_images(vfname, k, n, save_root)
