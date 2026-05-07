"""
Shared utilities for evaluation methods.

Provides:
  - load_frames: decode all frames from an mp4 into numpy arrays
  - is_static_video: return True if no pixels change across frames
  - find_changing_bbox: find the tightest square bounding box of pixels
                        that change across video frames
  - crop_frames: crop a region from a list of frames
  - compute_ssim: compute SSIM between two grayscale image arrays
  - mean_ssim_over_frames: average SSIM across aligned frame pairs
  - build_ssim_distance_matrix: full n×m SSIM distance matrix for DTW
  - clip_embeddings: extract L2-normalised CLIP embeddings per frame
  - build_clip_distance_matrix: full n×m cosine distance matrix via CLIP
"""

import cv2
import numpy as np

# Module-level model caches keyed by device string
_clip_cache: dict = {}
_dreamsim_cache: dict = {}


def load_frames(video_path: str) -> list[np.ndarray]:
    """Decode all frames from a video file.

    Returns a list of HxWx3 uint8 numpy arrays (RGB).
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads BGR; convert to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def is_static_video(frames: list[np.ndarray]) -> bool:
    """Return True if no pixels change across frames (video is static).

    Uses the same noise threshold as find_changing_bbox (>10 in any channel).
    """
    if len(frames) < 2:
        return True
    ref = frames[0].astype(np.int16)
    for frame in frames[1:]:
        diff = np.abs(frame.astype(np.int16) - ref)
        if diff.max(axis=2).max() > 10:
            return False
    return True


def find_changing_bbox(frames: list[np.ndarray], square: bool = True) -> tuple[int, int, int, int]:
    """Find the bounding box of pixels that change across all frames.

    Compares every frame against the first frame, builds a union mask of
    pixels that differ in any channel by more than a small threshold, then
    returns the tightest rectangle around that mask.

    Args:
        frames:  list of HxWx3 uint8 arrays.
        square:  if True, expand the rectangle into the tightest enclosing
                 square (centered on the rectangle).

    Returns:
        (x1, y1, x2, y2)  pixel coordinates (inclusive start, exclusive end).
        Returns full-frame bbox if no changing pixels are found.
    """
    if not frames:
        raise ValueError("frames list is empty")

    h, w = frames[0].shape[:2]
    ref = frames[0].astype(np.int16)

    # Accumulate a mask where any frame differs from the first frame
    change_mask = np.zeros((h, w), dtype=bool)
    for frame in frames[1:]:
        diff = np.abs(frame.astype(np.int16) - ref)
        # A pixel "changed" if any channel moved by more than noise threshold
        changed = diff.max(axis=2) > 10
        change_mask |= changed

    # Find rows/cols that contain at least one changed pixel
    rows = np.where(change_mask.any(axis=1))[0]
    cols = np.where(change_mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        # No motion detected – fall back to full frame
        return 0, 0, w, h

    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    x1, x2 = int(cols[0]), int(cols[-1]) + 1

    if not square:
        return x1, y1, x2, y2

    # Expand to the tightest enclosing square
    box_w = x2 - x1
    box_h = y2 - y1
    side = max(box_w, box_h)

    # Centre the square on the rectangle's centre
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    sx1 = max(0, cx - side // 2)
    sy1 = max(0, cy - side // 2)
    sx2 = sx1 + side
    sy2 = sy1 + side

    # Shift back if the square went past the frame boundary, then clamp
    if sx2 > w:
        sx1 -= sx2 - w
        sx2 = w
    if sy2 > h:
        sy1 -= sy2 - h
        sy2 = h
    sx1 = max(0, sx1)
    sy1 = max(0, sy1)

    return int(sx1), int(sy1), int(sx2), int(sy2)


def crop_frames(frames: list[np.ndarray], bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Crop each frame to the given (x1, y1, x2, y2) bounding box."""
    x1, y1, x2, y2 = bbox
    return [f[y1:y2, x1:x2] for f in frames]


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 uint8 RGB array to a float32 HxW array in [0, 1]."""
    # Standard luminance weights for RGB→gray
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return (frame @ weights) / 255.0


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute SSIM between two grayscale float32 arrays.

    Uses the standard SSIM formula with stabilisation constants
    c1 = (0.01)^2, c2 = (0.03)^2 (data range = 1.0).

    Args:
        a, b: 2-D float32 arrays of the same shape, values in [0, 1].

    Returns:
        SSIM scalar in [-1, 1].
    """
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(numerator / denominator)


def _resize_to_match(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize frame to target dimensions if they differ."""
    h, w = frame.shape[:2]
    if h == target_h and w == target_w:
        return frame
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def mean_ssim_over_frames(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    bbox: tuple[int, int, int, int] | None = None,
) -> tuple[float, list[float]]:
    """Compute per-frame SSIM between two frame lists, return mean and list.

    Truncates to the shorter of the two lists. Resizes frames_b to match
    frames_a dimensions if they differ.

    Args:
        frames_a: first list of HxWx3 uint8 RGB frames.
        frames_b: second list of HxWx3 uint8 RGB frames.
        bbox:     (x1, y1, x2, y2) crop region; None = full frame.

    Returns:
        (mean_score, per_frame_scores)
    """
    n = min(len(frames_a), len(frames_b))
    if n == 0:
        return 0.0, []

    target_h, target_w = frames_a[0].shape[:2]

    scores = []
    for i in range(n):
        fa = frames_a[i]
        fb = _resize_to_match(frames_b[i], target_h, target_w)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ca = to_grayscale(fa[y1:y2, x1:x2])
            cb = to_grayscale(fb[y1:y2, x1:x2])
        else:
            ca = to_grayscale(fa)
            cb = to_grayscale(fb)
        if ca.size == 0 or cb.size == 0:
            continue
        scores.append(compute_ssim(ca, cb))

    mean = float(np.mean(scores)) if scores else 0.0
    return mean, scores


# ─── DTW distance matrices ───────────────────────────────────────────────────


def build_ssim_distance_matrix(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    bbox: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Build a full n×m SSIM distance matrix for use with DTW.

    D[i, j] = 1 - SSIM(frames_a[i], frames_b[j]) cropped to bbox.
    Values are clamped to [0, 1].

    When device is not 'cpu', uses batched PyTorch operations on the GPU:
    the covariance term is computed via a single matrix multiply (A @ B.T),
    making the n×m build O(n*m*P) on CUDA rather than O(n*m) sequential CPU calls.

    Args:
        frames_a: first list of frames.
        frames_b: second list of frames.
        bbox:     (x1, y1, x2, y2) crop region; None = full frame.
        device:   torch device string ('cpu', 'cuda', 'cuda:0', …).

    Returns float32 array of shape (n, m).
    """
    n, m = len(frames_a), len(frames_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    if device != "cpu":
        return _build_ssim_distance_matrix_gpu(frames_a, frames_b, bbox, device)

    target_h, target_w = frames_a[0].shape[:2]

    D = np.empty((n, m), dtype=np.float32)
    for i, fa in enumerate(frames_a):
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ca = to_grayscale(fa[y1:y2, x1:x2])
        else:
            ca = to_grayscale(fa)
        for j, fb in enumerate(frames_b):
            fb_r = _resize_to_match(fb, target_h, target_w)
            if bbox is not None:
                cb = to_grayscale(fb_r[y1:y2, x1:x2])
            else:
                cb = to_grayscale(fb_r)
            if ca.size == 0 or cb.size == 0:
                D[i, j] = 1.0
            else:
                D[i, j] = max(0.0, 1.0 - compute_ssim(ca, cb))
    return D


def _build_ssim_distance_matrix_gpu(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    bbox: tuple[int, int, int, int] | None,
    device: str,
) -> np.ndarray:
    """GPU-batched SSIM distance matrix via PyTorch.

    Converts all frames to grayscale float32 tensors, then computes the full
    n×m SSIM matrix in one shot using a matrix multiply for the covariance term.
    """
    import torch

    target_h, target_w = frames_a[0].shape[:2]
    gray_w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32, device=device)

    def _to_flat(frame: np.ndarray, resize: bool = False) -> torch.Tensor:
        if resize:
            frame = _resize_to_match(frame, target_h, target_w)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = frame[y1:y2, x1:x2]
        t = torch.from_numpy(frame.astype(np.float32) / 255.0).to(device)  # H×W×3
        return (t @ gray_w).flatten()  # (P,)

    A = torch.stack([_to_flat(f) for f in frames_a])           # (n, P)
    B = torch.stack([_to_flat(f, resize=True) for f in frames_b])  # (m, P)
    P = float(A.shape[1])

    mu_a = A.mean(dim=1)   # (n,)
    mu_b = B.mean(dim=1)   # (m,)
    var_a = ((A - mu_a[:, None]) ** 2).mean(dim=1)  # (n,)
    var_b = ((B - mu_b[:, None]) ** 2).mean(dim=1)  # (m,)

    # Cov[i,j] = E[A_i * B_j] - mu_a[i]*mu_b[j];  E[A_i*B_j] = (A @ B.T) / P
    cov = (A @ B.T) / P - mu_a[:, None] * mu_b[None, :]  # (n, m)

    c1, c2 = 1e-4, 9e-4
    num = (2.0 * mu_a[:, None] * mu_b[None, :] + c1) * (2.0 * cov + c2)
    den = (mu_a[:, None] ** 2 + mu_b[None, :] ** 2 + c1) * (var_a[:, None] + var_b[None, :] + c2)

    D = torch.clamp(1.0 - num / den, min=0.0)
    return D.cpu().numpy().astype(np.float32)


# ─── CLIP utilities ───────────────────────────────────────────────────────────


def _get_clip(device: str = "cpu"):
    """Lazily load and cache a CLIP ViT-B/32 model and processor."""
    if device not in _clip_cache:
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        import torch
        model = model.to(device).eval()
        _clip_cache[device] = (model, processor)
    return _clip_cache[device]


def clip_embeddings(
    frames: list[np.ndarray],
    bbox: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Extract L2-normalised CLIP visual embeddings for each frame.

    Optionally crops each frame to bbox before embedding.

    Args:
        frames: list of HxWx3 uint8 RGB arrays.
        bbox:   (x1, y1, x2, y2) crop region; None = full frame.
        device: torch device string.

    Returns:
        (n, 512) float32 array of L2-normalised embeddings.
    """
    import torch
    from PIL import Image as PILImage

    model, processor = _get_clip(device)
    embeddings = []
    for frame in frames:
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                embeddings.append(np.zeros(512, dtype=np.float32))
                continue
            pil = PILImage.fromarray(crop)
        else:
            pil = PILImage.fromarray(frame)
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            feats = model.visual_projection(vision_out.pooler_output)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        embeddings.append(feats.cpu().numpy()[0])
    return np.array(embeddings, dtype=np.float32)


def check_video_fps(video_path: str, expected_fps: float = 30.0, tol: float = 0.5) -> bool:
    """Return True if the video's frame rate is within tol of expected_fps."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return abs(fps - expected_fps) <= tol


# ─── DreamSim utilities ───────────────────────────────────────────────────────


def _get_dreamsim(device: str = "cpu"):
    """Lazily load and cache a DreamSim model and preprocessor."""
    if device not in _dreamsim_cache:
        from dreamsim import dreamsim as load_dreamsim
        model, preprocess = load_dreamsim(pretrained=True, device=device)
        _dreamsim_cache[device] = (model, preprocess)
    return _dreamsim_cache[device]


def _dreamsim_embeddings(
    frames: list[np.ndarray],
    bbox: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
) -> "torch.Tensor":
    """Extract DreamSim embeddings for each frame. Returns a (n, D) CPU tensor."""
    import torch
    import torch.nn.functional as F
    from PIL import Image as PILImage

    model, preprocess = _get_dreamsim(device)

    embs = []
    _zero_emb = None
    for frame in frames:
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                if _zero_emb is not None:
                    embs.append(_zero_emb)
                else:
                    embs.append(None)
                continue
            pil = PILImage.fromarray(crop)
        else:
            pil = PILImage.fromarray(frame)
        t = preprocess(pil).to(device)  # (1, 3, 224, 224)
        with torch.no_grad():
            emb = model.embed(t)  # (1, D)
        if _zero_emb is None:
            _zero_emb = torch.zeros_like(emb).cpu()
        embs.append(emb.cpu())

    # Replace None (empty crop placeholders) with zero embeddings
    resolved = []
    for e in embs:
        if e is None:
            resolved.append(_zero_emb if _zero_emb is not None else torch.zeros(1, 768))
        else:
            resolved.append(e)
    return torch.cat(resolved, dim=0)  # (n, D)


def build_dreamsim_distance_matrix(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    bbox: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Build a full n×m DreamSim cosine-distance matrix for use with DTW.

    D[i, j] = 1 - cosine_similarity(embed_a[i], embed_b[j]), clamped to [0, 1].
    Lower = more similar (consistent with SSIM/CLIP distance matrices).

    Returns float32 array of shape (n, m).
    """
    import torch
    import torch.nn.functional as F

    n, m = len(frames_a), len(frames_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    emb_a = _dreamsim_embeddings(frames_a, bbox, device)  # (n, D)
    emb_b = _dreamsim_embeddings(frames_b, bbox, device)  # (m, D)

    # Cosine distance: 1 - cosine_similarity ∈ [0, 2]; clip to [0, 1]
    emb_a_norm = F.normalize(emb_a.float(), dim=-1)
    emb_b_norm = F.normalize(emb_b.float(), dim=-1)
    D = 1.0 - (emb_a_norm @ emb_b_norm.T)  # (n, m)
    return torch.clamp(D, min=0.0, max=1.0).numpy().astype(np.float32)


def mean_dreamsim_over_frames(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    bbox: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
) -> tuple[float, list[float]]:
    """Compute per-frame DreamSim similarity between two aligned frame lists.

    Uses model(img_a, img_b) per the official API, which returns a perceptual
    distance. Similarity = 1 - distance (higher is better, range ≈ [0, 1]).
    Truncates to the shorter list.

    Returns:
        (mean_score, per_frame_scores)
    """
    import torch
    from PIL import Image as PILImage

    n = min(len(frames_a), len(frames_b))
    if n == 0:
        return 0.0, []

    model, preprocess = _get_dreamsim(device)

    per_frame = []
    for fa, fb in zip(frames_a[:n], frames_b[:n]):
        def _prep(frame):
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                pil = PILImage.fromarray(crop) if crop.size > 0 else PILImage.fromarray(frame)
            else:
                pil = PILImage.fromarray(frame)
            return preprocess(pil).to(device)

        ta, tb = _prep(fa), _prep(fb)
        with torch.no_grad():
            dist = model(ta, tb)  # perceptual distance, lower = more similar
        per_frame.append(float(max(0.0, 1.0 - dist.item())))

    mean = float(np.mean(per_frame)) if per_frame else 0.0
    return mean, per_frame


def build_clip_distance_matrix(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    bbox: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Build a full n×m cosine distance matrix using CLIP embeddings for DTW.

    D[i, j] = 1 - cosine_similarity(frames_a[i], frames_b[j]) in bbox.
    Values are clamped to [0, 1].

    Returns float32 array of shape (n, m).
    """
    emb_a = clip_embeddings(frames_a, bbox, device)
    emb_b = clip_embeddings(frames_b, bbox, device)
    sim = emb_a @ emb_b.T  # (n, m)
    return np.maximum(0.0, 1.0 - sim).astype(np.float32)
