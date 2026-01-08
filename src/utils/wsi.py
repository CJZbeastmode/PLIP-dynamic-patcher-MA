import openslide
from PIL import Image
import torch
import math

# Import Embedder from the utils package. Use relative import when running as
# part of the package, fall back to absolute `src` path when modules are
# executed as scripts outside package context.
from .embedder import Embedder


class WSI:
    def __init__(
        self,
        image_path,
        patch_size=256,
        target_min_side=512,
        synthetic_scale=0.5,
        max_level_side=40000,
        embedder=None,
        img_embedding_backend="plip",
    ):
        """
        Load the WSI, initialize PLIP, generate synthetic pyramid levels.

        Native levels are kept as-is with their original scale factors.
        Synthetic levels are added at the coarse end with consistent 0.5 scale.
        """

        # ADDED COMMENT:
        # OpenSlide initialization is intentionally not wrapped in a try/except.
        # If this fails (e.g., corrupted file, unsupported format), the pipeline
        # cannot recover meaningfully and should fail fast.
        self.slide = openslide.OpenSlide(image_path)

        self.patch_size = patch_size
        self.target_min_side = target_min_side
        self.synthetic_scale = synthetic_scale
        self.max_level_side = max_level_side

        # Track level info
        self.levels_info = {}

        # Store synthetic images
        self.synthetic_images = {}

        # Load PLIP
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        if embedder is None:
            if img_embedding_backend == "plip":
                embedder = Embedder(img_backend="plip", device=device)
            elif img_embedding_backend == "conch":
                embedder = Embedder(img_backend="conch", device=device)
            else:
                embedder = Embedder()
        # Keep the Embedder instance for encoding convenience
        self.embedder = embedder
        self.model = embedder.model
        self.processor = embedder.processor

        self.device = device

        # Build native levels
        self._build_native_levels()

        self.min_level = min(
            lvl
            for lvl, info in self.levels_info.items()
            if not info.get("frozen", False)
        )

        # Generate synthetic levels at the coarse end
        self._generate_synthetic_levels()

        # ADDED COMMENT:
        # Explicit logging of level structure is useful for debugging
        # pyramid consistency and downstream coordinate transforms.
        for i in sorted(self.levels_info.keys()):
            info = self.levels_info[i]
            print(f"[{info['type']}] Level {i}: {info['size'][0]}x{info['size'][1]}")

    # =====================================================================
    # Build native levels (unchanged from original)
    # =====================================================================
    def _build_native_levels(self):
        """Add all native OpenSlide levels to levels_info."""
        self.min_level = 0  # will be updated

        for lvl in range(self.slide.level_count):
            w, h = self.slide.level_dimensions[lvl]

            frozen = max(w, h) > self.max_level_side

            if not frozen and lvl < getattr(self, "min_level", lvl):
                self.min_level = lvl

            pw = int(math.ceil(w / float(self.patch_size)) * self.patch_size)
            ph = int(math.ceil(h / float(self.patch_size)) * self.patch_size)

            self.levels_info[lvl] = {
                "size": (w, h),
                "padded_size": (pw, ph),
                "type": "native",
                "native_idx": lvl,
                "downsample": self.slide.level_downsamples[lvl],
                "frozen": frozen,
            }

    # =====================================================================
    # Generate synthetic coarse levels
    # =====================================================================
    def _generate_synthetic_levels(self):
        """
        Create downsampled synthetic levels beyond the coarsest native level.
        Uses consistent 0.5 scale factor for synthetic levels.
        """

        # Find the coarsest native level
        max_native_idx = self.slide.level_count - 1

        # If the coarsest native level is frozen, do not generate synthetic levels
        if self.levels_info[max_native_idx].get("frozen", False):
            self.max_level = max_native_idx
            return

        base_img = self.slide.read_region(
            (0, 0), max_native_idx, self.slide.level_dimensions[max_native_idx]
        ).convert("RGB")

        w, h = base_img.size
        scale = self.synthetic_scale
        current_level = max_native_idx + 1

        while True:
            new_w, new_h = int(w * scale), int(h * scale)

            # ADDED COMMENT:
            # Stop generating synthetic levels once resolution becomes
            # too small to yield meaningful patch-level features.
            if min(new_w, new_h) < self.target_min_side:
                break

            # Resize to create synthetic level
            synth = base_img.resize((new_w, new_h), Image.BILINEAR)

            # Pad synthetic image so its dimensions are multiples of patch_size
            pw = int(math.ceil(new_w / float(self.patch_size)) * self.patch_size)
            ph = int(math.ceil(new_h / float(self.patch_size)) * self.patch_size)
            if (pw, ph) != (new_w, new_h):
                canvas = Image.new("RGB", (pw, ph), (255, 255, 255))
                canvas.paste(synth, (0, 0))
                synth = canvas

            self.synthetic_images[current_level] = synth

            self.levels_info[current_level] = {
                "size": (new_w, new_h),
                "padded_size": (pw, ph),
                "type": "synthetic",
                "frozen": False,
            }

            base_img = synth
            w, h = new_w, new_h
            current_level += 1

        self.max_level = max(self.levels_info.keys())

    # =====================================================================
    # Patch extraction
    # =====================================================================
    def get_patch(self, lvl_id, x, y):
        """
        Extract patch from either real level or synthetic level.
        Raises exception if patch cannot be read (corrupted file).
        Output: img (PIL Image)
        """
        x = int(x)
        y = int(y)

        entry = self.levels_info[lvl_id]

        if entry.get("frozen", False):
            raise RuntimeError(
                f"Level {lvl_id} is frozen (size {entry['size']} exceeds limit)"
            )

        # use padded size for tiling / bounds when available
        if "padded_size" in entry:
            w, h = entry["padded_size"]
        else:
            w, h = entry["size"]

        # native WSI level
        if entry["type"] == "native":
            native_idx = entry["native_idx"]
            try:
                ds = self.slide.level_downsamples[native_idx]
                lx = int(x * ds)
                ly = int(y * ds)

                img = self.slide.read_region(
                    (lx, ly), native_idx, (self.patch_size, self.patch_size)
                ).convert("RGB")
                # Ensure patch is exactly patch_size x patch_size; pad with white if smaller
                if img.size != (self.patch_size, self.patch_size):
                    canvas = Image.new(
                        "RGB", (self.patch_size, self.patch_size), (255, 255, 255)
                    )
                    canvas.paste(img, (0, 0))
                    img = canvas
                return img
            except Exception as e:
                # ADDED COMMENT:
                # OpenSlide/OpenJPEG failures are surfaced explicitly
                # so callers (e.g. RL env) can catch and skip safely.
                raise RuntimeError(
                    f"Failed to read patch at level {lvl_id}, pos ({x},{y}): {e}"
                )

        # synthetic precomputed level
        elif entry["type"].lower() == "synthetic":
            img = self.synthetic_images[lvl_id]

            # Clamp start coordinates to padded image so cropping is safe.
            # This ensures patches that lie beyond original content are pure white.
            padded_w, padded_h = entry.get("padded_size") or entry["size"]
            max_x = max(0, padded_w - self.patch_size)
            max_y = max(0, padded_h - self.patch_size)
            x = max(0, min(x, max_x))
            y = max(0, min(y, max_y))

            x2 = x + self.patch_size
            y2 = y + self.patch_size

            patch = img.crop((x, y, x2, y2)).convert("RGB")
            # crop should be exactly patch_size due to padding, but keep safety pad
            if patch.size != (self.patch_size, self.patch_size):
                canvas = Image.new(
                    "RGB", (self.patch_size, self.patch_size), (255, 255, 255)
                )
                canvas.paste(patch, (0, 0))
                patch = canvas
            return patch

        else:
            raise ValueError(f"Unknown level type: {entry['type']}")

    # =====================================================================
    # Dynamic scale between levels
    # =====================================================================
    def get_scale(self, parent_level):
        """
        Compute scale factor from parent level to child (next finer) level.
        Scale = child_width / parent_width

        For native levels, this can be 2, 4, or other values.
        For synthetic levels, this is typically 2.
        """
        child_level = parent_level - 1
        if child_level < 0:
            return None

        pw, _ = self.levels_info[parent_level]["size"]
        cw, _ = self.levels_info[child_level]["size"]

        return cw / pw

    def get_num_children(self, parent_level):
        """
        Compute the number of child patches that fit in one parent patch.

        If scale = 2, we get 2x2 = 4 children.
        If scale = 4, we get 4x4 = 16 children.

        Returns (num_children_per_side, total_children)
        """
        scale = self.get_scale(parent_level)
        if scale is None:
            return None, None

        # Round to nearest integer for grid calculation
        num_per_side = int(round(scale))
        total = num_per_side * num_per_side

        return num_per_side, total

    def get_child_grid(self, parent_level, parent_x, parent_y):
        """
        Get the grid of child offsets for subdividing a parent patch.

        If `parent_x` and `parent_y` are not provided, this returns a list
        of (dx, dy) offsets in child-level pixels that describe the position
        of each child relative to the child's origin. This preserves the
        original behavior used by callers that only need offsets.

        If `parent_x` and `parent_y` are provided, this returns a list of
        child "grids" (one per child). Each child grid is itself a list of
        absolute (nx, ny) coordinates in the child-level pixel space that
        correspond to patches belonging to that child region. For the common
        case where the scale is 2 (2x2 children) each child grid will contain
        a single (nx, ny) coordinate (the child's top-left patch). For larger
        integer scales this returns one child entry per child (row-major).

        Returns:
            - if parent_x/y is None: List[(dx, dy)]
            - else: List[List[(nx, ny)]]  # one inner list per child
        """
        scale = self.get_scale(parent_level)
        if scale is None:
            return [] if parent_x is None else []

        num_per_side = int(round(scale))
        child_patch_size = self.patch_size  # Each child is patch_size x patch_size

        # Build offsets relative to a child's origin (child-level pixels)
        offsets = []
        for row in range(num_per_side):
            for col in range(num_per_side):
                dx = col * child_patch_size
                dy = row * child_patch_size
                offsets.append((dx, dy))

        # Otherwise compute absolute child-level coordinates for each offset.
        # parent_x,parent_y are expressed in parent-level pixels; convert to
        # child-level origin (cx,cy) using the scale factor cw/pw.
        child_level = parent_level - 1
        if child_level < 0:
            return []

        # compute child origin corresponding to parent's top-left
        cx = int(parent_x * scale)
        cy = int(parent_y * scale)

        # Group absolute coordinates per child (one entry per offset)
        child_grids = []
        for dx, dy in offsets:
            nx = cx + dx
            ny = cy + dy
            # currently each child grid contains a single top-left patch
            # coordinate; returning as a list allows extension later.
            child_grids.append([(nx, ny)])

        return child_grids

    # =====================================================================
    # Get embedding
    # =====================================================================
    def get_emb(self, img):
        """Get PLIP image embedding."""
        # Delegate to the central Embedder implementation which handles
        # PIL/numpy/torch inputs, blank-patch checks, normalization and caching.
        return self.embedder.img_emb(img)

    # =====================================================================
    # Patch iterator
    # =====================================================================
    def iterate_patches(self, lvl_id):
        """
        Generator that yields all patch coordinates for one level.
        """
        entry = self.levels_info[lvl_id]

        if entry.get("frozen", False):
            return

        if "padded_size" in entry:
            w, h = entry["padded_size"]
        else:
            w, h = entry["size"]

        for y in range(0, h, self.patch_size):
            for x in range(0, w, self.patch_size):
                yield x, y
