from torch.utils.data import Dataset
import os
import numpy as np
import math
import cv2


class AutoGenerateSegmentationDataset(Dataset):
    def __init__(
        self,
        path,
    ):
        self.path = path
        self.init_recipe()

    def init_recipe(self):
        self.base_size = 64
        self.use_hr_bg = True

        self.classes = [
            "enemy",
            "player",
            "tile",
        ]

        self.min_observer_num = 5
        self.max_observer_num = 20

    def generate(self):
        path_bg = os.path.join(self.path, "background")
        name_bg = "background.png"
        if self.use_hr_bg:
            name_bg = "background_HR.png"
        img = cv2.imread(os.path.join(path_bg, name_bg), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        label = np.zeros((h, w, len(self.classes)), dtype=np.uint8)

        img, label = self.generate_random_tile(img, label)
        img, label = self.generate_scourge(img, label)
        img = self.generate_none(img)
        img, label = self.generate_observer(img, label)

        cv2.imshow("preview", img)
        # for i in range(len(self.classes)):
        #     cv2.imshow(f"label_{self.classes[i]}", label[..., i])
        cv2.imshow("label", label)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return img, label

    def generate_observer(self, img, label):
        path_enemy = os.path.join(self.path, "enemy")
        names_enemy = os.listdir(path_enemy)

        observer_num = np.random.randint(self.min_observer_num, self.max_observer_num)

        h, w = img.shape[:2]
        labels = []
        for i in range(observer_num):
            name_enemy = np.random.choice(names_enemy)
            observer = cv2.imread(
                os.path.join(path_enemy, name_enemy), cv2.IMREAD_UNCHANGED
            )

            h_rand_pos = np.random.randint(0, h - self.base_size)
            w_rand_pos = np.random.randint(0, w - self.base_size)
            t = img[
                h_rand_pos : h_rand_pos + self.base_size,
                w_rand_pos : w_rand_pos + self.base_size,
            ]
            s = observer

            s_alpha = s[..., -1:] * 0.5

            patch = t[..., :-1] * (1 - s_alpha / 255) + s[..., :-1] * (s_alpha / 255)
            img[
                h_rand_pos : h_rand_pos + self.base_size,
                w_rand_pos : w_rand_pos + self.base_size,
                :-1,
            ] = patch

            t = label[
                h_rand_pos : h_rand_pos + self.base_size,
                w_rand_pos : w_rand_pos + self.base_size,
            ]
            now_label = np.zeros((h, w), dtype=label.dtype)
            now_label[
                h_rand_pos : h_rand_pos + self.base_size,
                w_rand_pos : w_rand_pos + self.base_size,
            ] = 255 * (s_alpha[..., 0] > 0)
            labels.append(now_label)

        label_max = np.max(np.stack(labels, axis=0), axis=0)
        label[..., self.classes.index("enemy")] = label_max
        return img, label

    def generate_none(self, img):
        path_none = os.path.join(self.path, "none")
        names_none = os.listdir(path_none)

        h, w = img.shape[:2]

        for name in names_none:
            h_rand_pos = np.random.randint(0, h - self.base_size)
            w_rand_pos = np.random.randint(0, w - self.base_size)
            none = cv2.imread(os.path.join(path_none, name), cv2.IMREAD_UNCHANGED)

            t = img[
                h_rand_pos : h_rand_pos + self.base_size,
                w_rand_pos : w_rand_pos + self.base_size,
            ]
            s = none

            patch = t[..., :-1] * (1 - s[..., -1:] / 255) + s[..., :-1] * (
                s[..., -1:] / 255
            )
            img[
                h_rand_pos : h_rand_pos + self.base_size,
                w_rand_pos : w_rand_pos + self.base_size,
                :-1,
            ] = patch

        return img

    def generate_scourge(self, img, label):
        path_player = os.path.join(self.path, "player")
        names_player = os.listdir(path_player)
        name_player = np.random.choice(names_player)

        h, w = img.shape[:2]

        h_half = h // 2 - (self.base_size // 2)
        w_half = w // 2 - (self.base_size // 2)

        rand_range = 10
        h_rand_pos = np.random.randint(-rand_range, rand_range)
        w_rand_pos = np.random.randint(-rand_range, rand_range)

        player = cv2.imread(
            os.path.join(path_player, name_player), cv2.IMREAD_UNCHANGED
        )
        y = h_half + h_rand_pos
        x = w_half + w_rand_pos
        mask = player[..., -1] > 0
        mask = mask[..., np.newaxis]

        patch = (
            img[y : y + self.base_size, x : x + self.base_size] * (1 - mask)
            + mask * player
        )

        img[y : y + self.base_size, x : x + self.base_size] = patch
        label[
            y : y + self.base_size, x : x + self.base_size, self.classes.index("player")
        ] = (mask[..., 0] * 255)

        return img, label

    def generate_random_tile(self, img, label):
        path_tile = os.path.join(self.path, "tile")
        names_tile = os.listdir(path_tile)
        h, w = img.shape[:2]

        size = 3
        h_rand = np.random.randint(0, h // size)
        w_rand = np.random.randint(0, w // size)

        h_res = h - h_rand
        w_res = w - w_rand

        h_tile_max_num = math.ceil(h_res / self.base_size)
        w_tile_max_num = math.ceil(w_res / self.base_size)

        h_tile_num = np.random.randint(int(h_tile_max_num * 0.75), h_tile_max_num)
        w_tile_num = np.random.randint(int(w_tile_max_num * 0.75), w_tile_max_num)

        for row in range(h_tile_num):
            for col in range(w_tile_num):
                name_tile = np.random.choice(names_tile)
                tile = cv2.imread(
                    os.path.join(path_tile, name_tile), cv2.IMREAD_UNCHANGED
                )
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2BGRA)

                y = row * self.base_size + h_rand
                x = col * self.base_size + w_rand

                now_h, now_w = img[
                    y : y + self.base_size, x : x + self.base_size
                ].shape[:2]

                img[y : y + self.base_size, x : x + self.base_size] = tile[
                    :now_h, :now_w
                ]
                label[
                    y : y + self.base_size,
                    x : x + self.base_size,
                    self.classes.index("tile"),
                ] = 255

        return img, label


if __name__ == "__main__":
    dataset = AutoGenerateSegmentationDataset(path=r"dataset")
    dataset.generate()
