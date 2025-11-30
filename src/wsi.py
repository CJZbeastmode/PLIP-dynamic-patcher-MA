import openslide
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


class WSI:
    def __init__(
        self,
        image_path,
        patch_size=256,
        target_min_side=512,
        synthetic_scale=0.5
    ):
        """
        Load the WSI, initialize PLIP, generate synthetic pyramid levels.
        """

        # Load WSI
        self.slide = openslide.OpenSlide(image_path)
        self.patch_size = patch_size
        self.target_min_side = target_min_side
        self.synthetic_scale = synthetic_scale

        # Track level info
        self.levels_info = {}

        # Store synthetic images separately
        self.synthetic_images = {}

        # Native / real levels
        for lvl in range(self.slide.level_count):
            self.levels_info[lvl] = {
                "size": self.slide.level_dimensions[lvl],
                "type": "native"
            }

        # Load PLIP
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("vinid/plip").to(device)
        self.processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.model.eval()
        self.device = device

        # Generate synthetic levels
        self._generate_synthetic_levels()

        for i in range(len(self.levels_info)):
            print(f"[{self.levels_info[i]['type']}] Level {i}: {self.levels_info[i]['size'][0]}x{self.levels_info[i]['size'][1]}")

    # =====================================================================
    # Synthetic pyramid generation
    # =====================================================================
    def _generate_synthetic_levels(self):
        """
        Create downsampled synthetic levels from the lowest native level.
        """

        lowest_native = self.slide.level_count - 1
        w, h = self.slide.level_dimensions[lowest_native]

        base_img = self.slide.read_region((0, 0), lowest_native, (w, h)).convert("RGB")
        scale = self.synthetic_scale
        current_level = lowest_native

        while True:
            new_w, new_h = int(w * scale), int(h * scale)

            if min(new_w, new_h) < self.target_min_side:
                break

            resized = base_img.resize((new_w, new_h), Image.BILINEAR)
            current_level += 1

            # Store synthetic level
            self.levels_info[current_level] = {
                "size": (new_w, new_h),
                "type": "synthetic"
            }
            self.synthetic_images[current_level] = resized

            w, h = new_w, new_h
            base_img = resized

        self.max_level = max(self.levels_info.keys())

    # =====================================================================
    # Patch extraction
    # =====================================================================
    def get_patch(self, lvl_id, x, y):
        """
        Extract patch from either real level or synthetic level.
        """
        x = int(x)
        y = int(y)

        entry = self.levels_info[lvl_id]
        w, h = entry["size"]

        # native WSI level
        if entry["type"] == "native":
            ds = self.slide.level_downsamples[lvl_id]
            lx = int(x * ds)
            ly = int(y * ds)

            return self.slide.read_region(
                (lx, ly),
                lvl_id,
                (self.patch_size, self.patch_size)
            ).convert("RGB")

        # synthetic precomputed level
        elif entry["type"].lower() == "synthetic":
            img = self.synthetic_images[lvl_id]

            # coordinate clamping
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            x2 = min(x + self.patch_size, w)
            y2 = min(y + self.patch_size, h)

            if x2 <= x or y2 <= y:
                raise ValueError(f"Patch outside bounds at level {lvl_id}: ({x}, {y})")

            return img.crop((x, y, x2, y2))
        
        else:
            raise ValueError(f"Unknown level type: {type}")


    # =====================================================================
    # Scale between levels
    # =====================================================================
    def get_scale(self, parent_level):
        """
        Compute scale factor to the child (next finer) level.
        Scale = parent_width / child_width
        """

        child_level = parent_level - 1
        if child_level < 0:
            return None

        def get_dims(lvl):
            if self.levels_info[lvl]["type"] == "native":
                return self.slide.level_dimensions[lvl]
            return self.levels_info[lvl]["size"]

        pw, _ = get_dims(parent_level)
        cw, _ = get_dims(child_level)

        return pw / cw

    # =====================================================================
    # Get embedding
    # =====================================================================
    def get_emb(self, img):
        """Get PLIP image embedding."""
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray((img * 255).astype("uint8"))
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img = img.convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            e = self.model.get_image_features(**inputs)
        return e / e.norm()

    # =====================================================================
    # Patch iterator
    # =====================================================================
    def iterate_patches(self, lvl_id):
        """
        Generator that yields all patch coordinates for one level.
        """
        w, h = self.levels_info[lvl_id]["size"]

        for y in range(0, h, self.patch_size):
            for x in range(0, w, self.patch_size):
                yield x, y
