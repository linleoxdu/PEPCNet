from monai.utils.enums import TransformBackends
from monai.transforms.transform import RandomizableTransform, Transform, MapTransform
import numpy as np
import torch
import torch.nn.functional as F
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor
from typing import Any, Hashable, Mapping


class StimulateLowResolutionTransform(Transform):

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        factor: float | None = None,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        """
        Args:
            minv: minimum value of output data.
            maxv: maximum value of output data.
            factor: factor scale by ``v = v * (1 + factor)``. In order to use
                this parameter, please set both `minv` and `maxv` into None.
            channel_wise: if True, scale on each channel separately. Please ensure
                that the first dimension represents the channel of the image if True.
            dtype: output data type, if None, same as input image. defaults to float32.
        """
        self.factor = factor
        self.channel_wise = channel_wise
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.

        Raises:
            ValueError: When ``self.minv=None`` or ``self.maxv=None`` and ``self.factor=None``. Incompatible values.

        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = convert_to_tensor(img, track_meta=False)
        ret: NdarrayOrTensor

        row_size = list(img.shape[-3:])
        target_size = [round(s * self.factor) for s in row_size]

        ret = F.interpolate(torch.unsqueeze(img_t, dim=0), size=target_size, mode="nearest")
        ret = torch.squeeze(F.interpolate(ret, size=row_size, mode="trilinear"), dim=0)

        ret = convert_to_dst_type(ret, dst=img, dtype=self.dtype or img_t.dtype)[0]
        return ret


class RandStimulateLowResolutionTransform(RandomizableTransform):

    backend = StimulateLowResolutionTransform.backend

    def __init__(self, factors: tuple[float, float] | float, prob: float = 0.1, dtype: DtypeLike = np.float32) -> None:
        """
        Args:
            factors: factor range to randomly scale by ``v = v * (1 + factor)``.
                if single number, factor value is picked from (-factors, factors).
            prob: probability of scale.
            dtype: output data type, if None, same as input image. defaults to float32.

        """
        RandomizableTransform.__init__(self, prob)
        if isinstance(factors, (int, float)):
            self.factors = (min(-factors, factors), max(-factors, factors))
        elif len(factors) != 2:
            raise ValueError(f"factors should be a number or pair of numbers, got {factors}.")
        else:
            self.factors = (min(factors), max(factors))
        self.factor = self.factors[0]
        self.dtype = dtype

    def randomize(self, data: Any | None = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.factor = self.R.uniform(low=self.factors[0], high=self.factors[1])

    def __call__(self, img: NdarrayOrTensor, randomize: bool = True) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        if randomize:
            self.randomize()

        if not self._do_transform:
            return convert_data_type(img, dtype=self.dtype)[0]

        return StimulateLowResolutionTransform(factor=self.factor, dtype=self.dtype)(img)


class RandStimulateLowResolutionTransformd(RandomizableTransform, MapTransform):

    backend = RandStimulateLowResolutionTransform.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob: float,
        factors: tuple[float, float] | float,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_stimulate_low_resolution = RandStimulateLowResolutionTransform(factors, prob=prob, dtype=dtype)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ):
        super().set_random_state(seed, state)
        self.rand_stimulate_low_resolution.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # all the keys share the same random noise
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.rand_stimulate_low_resolution.randomize(d[first_key])

        for key in self.key_iterator(d):
            d[key] = self.rand_stimulate_low_resolution(img=d[key], randomize=False)
        return d