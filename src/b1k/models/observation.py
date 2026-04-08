"""Observation 자료형과 전처리 함수 (observation.py)

이 파일의 역할:
    모델이 한 스텝에서 받는 모든 입력(관측값)을 하나의 구조체로 묶는다.
    openpi의 기본 Observation에 B1K/FAST 전용 필드를 추가한 확장 버전이다.

주요 설계 결정:
    1. flax.struct.dataclass 사용: JAX pytree로 자동 등록되어 jit/vmap에서 투명하게 작동한다.
    2. Generic[ArrayT]: JAX array, NumPy array, PyTorch tensor 등 다양한 배열 타입을 지원한다.
    3. 이미지는 [-1, 1] 범위: SigLIP/PaliGemma의 표준 입력 범위에 맞춘다.
    4. FAST 필드는 Optional: use_fast_auxiliary=False이면 None으로 두어 메모리를 낭비하지 않는다.
"""

from collections.abc import Sequence
from typing import Generic, TypeVar
import dataclasses

import augmax          # JAX 기반 이미지 증강 라이브러리
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.shared import image_tools
from openpi.shared import array_typing as at

# 배열 타입 제네릭 변수: JAX array, NumPy array, PyTorch tensor 등을 모두 허용한다.
# 이렇게 하면 학습 중(JAX array)과 추론 중(NumPy → JAX 변환 전)에 같은 자료형을 쓸 수 있다.
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)

# 이미지 키 이름 (데이터셋 키 이름과 일치해야 한다)
IMAGE_KEYS = (
    "base_0_rgb",         # 로봇 base 카메라 (정면 전경 뷰)
    "left_wrist_0_rgb",   # 왼손 손목 카메라 (근접 조작 뷰)
    "right_wrist_0_rgb",  # 오른손 손목 카메라 (근접 조작 뷰)
)

# SigLIP/PaliGemma 입력 해상도. 이 크기로 resize_with_pad를 적용한다.
IMAGE_RESOLUTION = (224, 224)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Observation 자료형
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """모델 한 스텝 입력을 모두 담는 구조체.

    @struct.dataclass: flax가 이 클래스를 JAX pytree로 등록한다.
        → jax.jit, jax.vmap, jax.grad 등에서 자동으로 분해/재조립된다.
        → 학습 루프에서 jit(train_step)(obs, actions) 형태로 바로 쓸 수 있다.

    @at.typecheck: 배열 shape/dtype을 런타임에 검증한다.
        → 잘못된 shape로 모델을 돌리다가 나중에 에러 나는 것을 방지한다.

    Generic[ArrayT]: JAX / NumPy / PyTorch 배열을 모두 수용한다.
        → 데이터 로더(NumPy)와 모델 forward(JAX array) 간에 같은 자료형을 쓸 수 있다.

    필드 설명:
        images:             카메라별 RGB 이미지. 범위는 [-1, 1]로 정규화되어야 한다.
                            shape: [*batch, H, W, C]
        image_masks:        해당 카메라 이미지가 실제로 존재하는지 나타내는 boolean 마스크.
                            shape: [*batch]
        state:              로봇 고유감각(관절 각도, 속도, 그리퍼 상태 등).
                            shape: [*batch, state_dim]
        tokenized_prompt:   태스크/stage 정보를 담은 정수 배열.
                            기본 경로: [로컬 task_id]  (shape[1]=1)
                            stage 경로: [로컬 task_id, stage_id] (shape[1]=2)
        tokenized_prompt_mask: tokenized_prompt의 어느 위치가 유효한지 나타내는 마스크.
        token_ar_mask:      자기회귀(autoregressive) 마스크. 1이면 앞 토큰만 볼 수 있다.
        token_loss_mask:    손실 계산에 포함할 위치. 1인 위치만 손실에 기여한다.
        fast_tokens:        FAST 보조 학습용 토큰 시퀀스 (use_fast_auxiliary=True일 때만 사용).
                            shape: [*batch, max_fast_tokens]
        fast_token_mask:    fast_tokens 중 실제로 유효한 위치 마스크.
    """

    # ── 필수 필드 ──────────────────────────────────────────────────────────
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]

    # ── 선택 필드 (기본값 None) ────────────────────────────────────────────
    # tokenized_prompt: 태스크/stage 정보. None이면 task 조건 없이 동작한다.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None

    # tokenized_prompt_mask: prompt의 유효 위치 마스크.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # token_ar_mask: 각 토큰의 자기회귀 여부. prefix 구성에 사용된다.
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None

    # token_loss_mask: 손실 계산 대상 위치. FAST 학습 시 target 위치를 나타낸다.
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # FAST 보조 학습 전용 필드 ────────────────────────────────────────────────
    # fast_tokens: FAST 토크나이저가 생성한 action 시퀀스 토큰 id.
    #   None이면 use_fast_auxiliary=False 이거나 해당 샘플에 FAST 토큰이 없는 것이다.
    fast_tokens: at.Int[ArrayT, "*b t"] | None = None

    # fast_token_mask: fast_tokens 중 패딩이 아닌 실제 토큰 위치.
    fast_token_mask: at.Bool[ArrayT, "*b t"] | None = None

    # ── 클래스 메서드 ──────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """딕셔너리에서 Observation 인스턴스를 만든다.

        데이터 로더가 반환하는 딕셔너리 형태를 Observation 구조체로 변환할 때 사용한다.

        전처리:
            - uint8 이미지를 float32 [-1, 1] 범위로 자동 변환한다.
              (SigLIP/PaliGemma의 표준 입력 범위)
            - tokenized_prompt / tokenized_prompt_mask는 둘 다 있거나 둘 다 없어야 한다.

        Args:
            data: 데이터 로더에서 반환된 딕셔너리.
                필수 키: "image", "image_mask", "state"
                선택 키: "tokenized_prompt", "tokenized_prompt_mask",
                         "token_ar_mask", "token_loss_mask",
                         "fast_tokens", "fast_token_mask"

        Returns:
            Observation 인스턴스.

        Raises:
            ValueError: tokenized_prompt와 tokenized_prompt_mask 중 하나만 있을 때.
        """
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError(
                "tokenized_prompt와 tokenized_prompt_mask는 반드시 함께 제공해야 합니다. "
                "둘 다 있거나 둘 다 없어야 합니다."
            )

        # uint8 이미지를 float32 [-1, 1] 범위로 변환한다.
        # SigLIP/PaliGemma 모델이 이 범위를 기대하기 때문이다.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                # NumPy uint8: [0, 255] → float32 [-1, 1]
                # 공식: value / 255.0 * 2.0 - 1.0
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                # PyTorch uint8: 채널 순서가 [H, W, C]가 아닐 수 있으므로 permute 후 변환.
                data["image"][key] = (
                    data["image"][key]
                    .to(torch.float32)
                    .permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
                    / 255.0 * 2.0 - 1.0
                )

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            fast_tokens=data.get("fast_tokens"),
            fast_token_mask=data.get("fast_token_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Observation을 딕셔너리로 변환한다.

        from_dict의 역방향 변환이다.
        직렬화, 로깅, 혹은 openpi의 구 코드와 호환할 때 사용한다.
        """
        result = dataclasses.asdict(self)
        # 필드 이름을 데이터 로더 규약(image, image_mask)에 맞게 변환한다.
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전처리 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """이미지를 224×224로 resize하고, 학습 시에는 데이터 증강을 적용한다.

    전처리 순서:
        1. resize_with_pad: 원본 해상도 → 224×224 (비율 보존, 패딩 추가)
        2. (train=True만) augmax 증강:
            - base 카메라: RandomCrop(95%) → Resize → Rotate(±5°) → ColorJitter
            - wrist 카메라: ColorJitter만 (손목 카메라는 조작 세부 정보가 중요하므로 기하 변환 제외)
        3. FAST 필드는 변환하지 않고 그대로 보존한다.

    Args:
        rng: JAX 난수 키. train=True일 때 증강에 사용된다. train=False이면 None 가능.
        observation: 전처리할 Observation. 이미지는 [-1, 1] 범위여야 한다.
        train: True이면 데이터 증강을 적용한다. False이면 resize만 한다.
        image_keys: 전처리할 이미지 키 목록.
        image_resolution: 목표 해상도 (H, W).

    Returns:
        전처리된 새 Observation 인스턴스. 원본은 변경되지 않는다.

    Raises:
        ValueError: observation.images에 image_keys 중 일부가 없을 때.
    """
    # 필요한 이미지 키가 모두 있는지 확인한다.
    if not set(image_keys).issubset(observation.images):
        raise ValueError(
            f"observation.images에 필요한 키가 없습니다. "
            f"기대: {image_keys}, 실제: {list(observation.images)}"
        )

    batch_shape = observation.state.shape[:-1]  # 배치 차원 (state의 마지막 dim 제외)

    out_images = {}
    for key in image_keys:
        image = observation.images[key]  # [*batch, H, W, C], float32 [-1, 1]

        # 1. Resize: 원본 해상도가 target과 다를 때만 수행한다.
        if image.shape[1:3] != image_resolution:
            # resize_with_pad: 비율을 유지하면서 패딩을 추가해 정확히 target 크기로 맞춘다.
            # 단순 stretch resize는 물체 비율을 왜곡하므로 사용하지 않는다.
            image = image_tools.resize_with_pad(image, *image_resolution)

        # 2. 데이터 증강 (학습 시에만 적용)
        if train:
            # augmax는 [0, 1] 범위 이미지를 기대하므로 잠시 변환한다.
            # 변환 공식: augmax_input = (model_input + 1) / 2
            image = image / 2.0 + 0.5  # [-1, 1] → [0, 1]

            transforms = []

            # wrist 카메라가 아닌 경우(base 카메라): 기하학적 변환을 추가한다.
            # wrist 카메라는 손 근처를 보는 카메라이므로,
            # 강한 기하 변환 시 조작 대상의 세부 정보가 사라질 위험이 있다.
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    # 95% 크기로 랜덤 crop → 카메라 위치 변화 시뮬레이션
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    # 원래 크기로 다시 resize
                    augmax.Resize(width, height),
                    # ±5° 범위 랜덤 회전 → 로봇 자세 변화 시뮬레이션
                    augmax.Rotate((-5, 5)),
                ]

            # 모든 카메라에 색상 변환을 추가한다.
            # 밝기/대비/채도 변화로 조명 변화에 강건하게 만든다.
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]

            # 배치 내 각 샘플에 독립적인 난수 키를 부여해 vmap으로 병렬 증강한다.
            # jax.vmap으로 배치 내 각 이미지를 독립적으로 변환한다.
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 증강 완료 후 다시 [-1, 1] 범위로 복원한다.
            # 모델이 이 범위를 기대하기 때문이다.
            image = image * 2.0 - 1.0  # [0, 1] → [-1, 1]

        out_images[key] = image

    # 이미지 마스크를 만든다.
    # 원본 observation에 마스크가 없으면 "이미지가 항상 존재"하는 것으로 간주한다.
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 마스크 없음 → 모든 위치가 유효(True)인 마스크 생성
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)
        else:
            # 원본 마스크를 JAX array로 변환 (NumPy나 Python bool일 수도 있으므로)
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    # 새로운 Observation을 만들어 반환한다.
    # 이미지와 마스크만 업데이트하고, 나머지 필드는 원본 그대로 보존한다.
    # 특히 fast_tokens, fast_token_mask는 이미지와 무관하므로 변경 없이 유지한다.
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        # getattr로 접근하는 이유: 오래된 체크포인트나 외부 코드에서 만든
        # Observation에는 fast 필드가 없을 수도 있기 때문이다.
        fast_tokens=getattr(observation, 'fast_tokens', None),
        fast_token_mask=getattr(observation, 'fast_token_mask', None),
    )
