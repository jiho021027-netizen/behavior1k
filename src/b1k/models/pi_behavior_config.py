"""BEHAVIOR-1K 모델 설정 파일 (pi_behavior_config.py)

이 파일은 PiBehavior 모델 전체의 하이퍼파라미터와 구조적 선택을 한 곳에 모아 둔다.

설계 원칙:
    1. backbone(PaliGemma / SigLIP)은 pi0 / pi05_base 웨이트를 그대로 가져온다.
       → vision-language 사전 학습 능력을 보존하면서 fine-tuning 비용을 줄인다.
    2. task embedding / stage head 등 B1K 전용 파라미터는 새로 랜덤 초기화한다.
       → pi0 체크포인트에는 없는 가중치이므로 weight_loaders.py에서 skip 처리한다.
    3. 실제 학습/평가에서 사용하는 태스크는 50개 전체가 아니라 선택한 12개 subset이다.
       → num_tasks=12 로 embedding 표 크기를 줄여 메모리와 overfitting을 절약한다.
    4. flow matching 방식으로 action을 예측한다.
       → DDPM 대신 linear interpolation(flow) 기반 diffusion을 써서 추론 단계를 줄인다.
    5. task embedding + stage conditioning 구조는 남겨 두어
       나중에 50-task로 확장하거나 multi-stage 학습을 할 때 재활용할 수 있게 한다.
"""

import dataclasses
import json
import pathlib
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from b1k.models.observation import Observation

# TYPE_CHECKING: 순환 import를 피하기 위해 타입 힌트 전용으로만 import한다.
# 실행 시점에는 이 블록이 실행되지 않으므로 실제 import 비용이 없다.
if TYPE_CHECKING:
    from b1k.models.pi_behavior import PiBehavior


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task / Stage 메타데이터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 선택한 12개 태스크 각각에 몇 개의 stage(subtask)가 있는지 정의한다.
# 인덱스 순서는 task_subset.py 의 SELECTED_TASKS 순서와 반드시 동일해야 한다.
#
# 예:
#   로컬 id 0 → stage 5개   (SELECTED_TASKS[0] = 전역 id 0번 태스크)
#   로컬 id 1 → stage 6개   (SELECTED_TASKS[1] = 전역 id 1번 태스크)
#   ...
#
# 이 값은 아래에서 두 가지 목적으로 사용된다.
#   1) encode_subtask_state(): stage를 [0, 1] 범위로 정규화할 때 분모
#   2) compute_detailed_loss(): valid stage mask를 만들 때 태스크별 stage 수 확인
TASK_NUM_STAGES: tuple[int, ...] = (5, 6, 15, 12, 14, 15, 13, 13, 12, 14, 15, 9)

# 모든 태스크를 통틀어 가장 많은 stage 수.
# stage_pred_from_vlm 분류 head의 출력 차원(클래스 수)이 이 값과 같아야 한다.
# 태스크마다 실제 stage 수가 다르기 때문에, 해당 태스크에 없는 stage 위치는
# forward pass에서 -inf masking으로 처리해 softmax에서 확률이 0이 되도록 한다.
MAX_NUM_STAGES: int = 15

# task_stage_embeddings Embed 테이블의 전체 크기.
# 12개 태스크의 stage 수를 모두 더한 값이다.
# 각 (task, stage) 쌍이 embedding 테이블 안에서 고유한 행을 가진다.
# → 태스크를 가로질러 "같은 번호 stage"가 같은 embedding을 공유하지 않도록 설계한다.
#   (예: 태스크 0의 stage 2 ≠ 태스크 1의 stage 2)
TOTAL_TASK_STAGE_EMBEDDINGS: int = sum(TASK_NUM_STAGES)  # = 143

# 태스크별 stage embedding 시작 오프셋.
# task i의 stage j는 embedding 인덱스 TASK_STAGE_OFFSETS[i] + j 에 해당한다.
#
# 예: TASK_NUM_STAGES = (5, 6, 15, ...)
#   TASK_STAGE_OFFSETS = (0, 5, 11, 26, ...)
#   → 태스크 0의 stage 0~4 → embed index 0~4
#   → 태스크 1의 stage 0~5 → embed index 5~10
#   → 태스크 2의 stage 0~14 → embed index 11~25
TASK_STAGE_OFFSETS: tuple[int, ...] = tuple(
    [0] + [sum(TASK_NUM_STAGES[:i + 1]) for i in range(len(TASK_NUM_STAGES) - 1)]
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 설정 dataclass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclasses.dataclass(frozen=True)
class PiBehaviorConfig(_model.BaseModelConfig):
    """PiBehavior 모델의 모든 하이퍼파라미터를 담는 불변(frozen) dataclass.

    frozen=True 이기 때문에 인스턴스를 만든 뒤에는 필드 값을 직접 바꿀 수 없다.
    설정을 바꾸려면 dataclasses.replace(config, field=new_value) 로 새 인스턴스를 만든다.

    전체 모델 구조 요약:
        ┌──────────────────────────────────────────────────────────┐
        │  [PREFIX]                                                │
        │    SigLIP vision encoder  → 이미지 토큰 (224×224 → 256개) │
        │    PaliGemma (2B Gemma)   → 이미지/태스크/상태 처리       │
        │    task_embeddings        → 12개 태스크별 학습 벡터       │
        │    task_stage_embeddings  → (task, stage) 쌍별 벡터      │
        ├──────────────────────────────────────────────────────────┤
        │  [SUFFIX - flow matching]                                │
        │    Action Expert (300M)   → 노이즈 action 복원           │
        │    flow matching scheduler → linear ODE 기반 샘플링      │
        └──────────────────────────────────────────────────────────┘

    Pi0와의 주요 차이점:
        - 텍스트 프롬프트 대신 num_tasks개의 학습 가능한 task embedding 사용
        - stage conditioning 지원 (tokenized_prompt = [task_id, stage_id])
        - pi0 체크포인트에서 backbone 가중치를 가져오고 신규 파라미터만 재초기화
    """

    # ── 기본 dtype ──────────────────────────────────────────────────────────
    # bfloat16: float16과 같은 메모리(2bytes)를 쓰지만 float32와 동일한 지수 범위를 가진다.
    # → 수치적으로 더 안정적이고 TPU/A100에서 native 지원된다.
    dtype: str = "bfloat16"

    # ── backbone 변형 ────────────────────────────────────────────────────────
    # paligemma_variant: VLM(시각-언어 모델) prefix 부분의 Gemma 크기.
    #   "gemma_2b" → 약 20억 파라미터. 이미지 토큰 + 태스크 토큰을 처리한다.
    #   이 부분은 pi0 체크포인트에서 가중치를 그대로 가져온다.
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # action_expert_variant: action 예측만을 담당하는 소형 Gemma.
    #   "gemma_300m" → 약 3억 파라미터. 노이즈 섞인 action → clean action 복원.
    #   adaRMS conditioning으로 diffusion timestep 정보를 받는다.
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # ── action 관련 치수 ──────────────────────────────────────────────────────
    # action_dim: 한 타임스텝 action 벡터의 차원 수.
    #   B1K 로봇 기준 32차원 (팔 6DOF × 2 + 손가락 + 그리퍼 등).
    action_dim: int = 32

    # action_horizon: 한 번에 예측하는 미래 타임스텝 수.
    #   30 → 30Hz 기준 약 1초 분량의 미래 action을 한 번에 예측.
    #   이 값이 커질수록 더 먼 미래를 예측하지만 계산량도 증가한다.
    action_horizon: int = 30

    # max_token_len: 텍스트 토크나이저 호환성을 위해 남겨 두는 값.
    #   실제 B1K 모델은 텍스트 프롬프트를 쓰지 않으므로 forward pass에서 사용되지 않는다.
    max_token_len: int = 200

    # ── 태스크 embedding 설정 ────────────────────────────────────────────────
    # num_tasks: task_embeddings 테이블의 행 수 = 학습할 태스크 수.
    #   Pi0 원본은 50개지만, 이번 실험에서는 선택한 12개만 사용한다.
    #   이 값을 줄이면:
    #     - 메모리 절약 (12 × 2048 × 2bytes ≈ 48KB, 무시할 수준이지만 명확성을 위해)
    #     - 학습 시 gradient가 실제 사용하는 embedding만 업데이트됨
    #     - 나중에 num_tasks를 늘리면 기존 12개 embedding을 보존하면서 확장 가능
    #   ⚠ 주의: 이 값을 바꾸면 pi0 체크포인트의 task_embeddings shape와 달라진다.
    #          weight_loaders.py에서 task_embeddings는 재초기화(missing)로 처리한다.
    num_tasks: int = 12

    # task_embedding_dim: 태스크 embedding 벡터의 차원.
    #   None이면 __post_init__에서 paligemma_variant의 hidden width로 자동 설정된다.
    #   (gemma_2b의 경우 width=2048)
    #   PaliGemma hidden dim과 같게 맞추면 별도 projection 없이 prefix token으로 바로 쓸 수 있다.
    task_embedding_dim: int = None  # type: ignore

    # max_num_subtask_states: stage prediction head 출력 클래스 수 (= MAX_NUM_STAGES).
    max_num_subtask_states: int = MAX_NUM_STAGES

    # task_data_path: task 이름/설명이 담긴 JSON 파일 경로.
    #   현재 forward pass에서는 직접 사용하지 않지만, 디버그나 시각화용으로 보존한다.
    task_data_path: str = "b1k/BEHAVIOR-1K/docs/challenge/task_data.json"

    # ── Correlated noise 설정 ────────────────────────────────────────────────
    # use_correlated_noise: action들 사이의 시간적 상관관계를 반영한 noise를 쓸지 여부.
    #   기본 Pi0는 독립 가우시안 noise를 사용한다.
    #   True이면 norm_stats 안에 action_correlation_cholesky 행렬이 반드시 있어야 한다.
    #   (compute_norm_stats.py --correlation 옵션으로 미리 계산해야 한다.)
    #
    #   correlated noise를 쓰면:
    #     - noise가 실제 action 분포의 시간적 패턴을 반영한다.
    #     - 특히 저주파 동작(느린 팔 이동)에서 flow matching이 더 쉬워질 수 있다.
    #     - 단, ill-conditioned 공분산 행렬은 학습 불안정을 유발할 수 있다.
    use_correlated_noise: bool = False

    # correlation_beta: 공분산 행렬 shrinkage 파라미터.
    #   실측 공분산 행렬이 노이즈나 outlier 때문에 ill-conditioned할 수 있으므로,
    #   항등행렬 I와 선형 결합해서 정칙화(regularize)한다.
    #     Σ_reg = beta × Σ_measured + (1 - beta) × I
    #   beta=1.0 → 실측 공분산만 사용 (불안정할 수 있음)
    #   beta=0.5 → 실측 50% + 독립 50% 혼합 (기본값, 권장)
    #   beta=0.0 → 완전 독립 noise (use_correlated_noise=False와 동일)
    correlation_beta: float = 0.5

    # ── FAST 보조 학습 설정 ───────────────────────────────────────────────────
    # FAST(Frequency-domain Action Sequence Tokenization):
    #   action 시퀀스를 주파수 도메인 토큰으로 압축해 VLM prefix에서
    #   자기회귀(autoregressive) 방식으로 예측하는 보조 학습 기법.
    #   VLM의 시퀀스 생성 능력을 action 예측에 직접 활용하려는 시도이다.

    # use_fast_auxiliary: 학습 시 FAST 보조 손실을 켤지 여부.
    #   True이면:
    #     - fast_token_embedding, fast_token_proj 레이어가 모델에 추가된다.
    #     - prefix에 FAST 토큰 시퀀스가 포함된다 (teacher forcing 방식).
    #     - 총 손실 = flow_loss + fast_loss_weight × fast_cross_entropy_loss
    use_fast_auxiliary: bool = False

    # fast_loss_weight: 전체 손실에서 FAST 손실이 차지하는 비중.
    #   0.1(기본)로 작게 설정해 FAST가 flow matching 학습을 방해하지 않게 한다.
    fast_loss_weight: float = 0.1

    # fast_encoded_dims: FAST가 압축할 action 차원의 범위 (문자열 또는 리스트).
    #   "0:6,7:23" → 0~5번(팔 관절 6DOF)과 7~22번(손 관절 16DOF)을 FAST로 인코딩.
    #   나머지 차원(그리퍼 이진 신호 등)은 FAST에서 제외한다.
    fast_encoded_dims: str | list[tuple[int, int]] = "0:6,7:23"

    # fast_vocab_size: FAST 토크나이저의 어휘 크기 (codebook 크기).
    #   1024 → action 시퀀스를 1024가지 이산 토큰 중 하나로 매핑한다.
    fast_vocab_size: int = 1024

    # max_fast_tokens: FAST가 생성하는 최대 토큰 수.
    #   action_horizon=30 action을 압축하면 보통 10~30개 토큰이 나오는데,
    #   이 값을 넘으면 앞에서부터 잘라낸다.
    max_fast_tokens: int = 32

    # fast_tokenizer_path: 사전 학습된 FAST 토크나이저 파일 경로.
    #   None이면 train_fast_tokenizer.py로 먼저 학습해야 한다.
    fast_tokenizer_path: str | None = None

    # ── KV cache 변환 설정 ────────────────────────────────────────────────────
    # use_kv_transform: KVCacheTransform 모듈을 사용할지 여부.
    #   기본 Pi0는 VLM layer i의 KV cache를 action expert layer i가 1:1로 참조한다.
    #   True이면 각 action expert layer가 여러 VLM layer의 KV cache를 가중합해서 본다.
    #   → 더 유연하지만 파라미터가 늘어나므로 기본값은 False.
    use_kv_transform: bool = False

    # ── Knowledge insulation 설정 ─────────────────────────────────────────────
    # use_knowledge_insulation: action expert → VLM 방향의 역전파를 차단하는 옵션.
    #   True이면:
    #     - VLM 본체는 FAST 손실(언어 방향)로만 업데이트된다.
    #     - flow matching 손실은 action expert와 task/stage head만 업데이트한다.
    #   VLM을 고정하고 task/stage head와 action expert만 학습하고 싶을 때 유용하다.
    use_knowledge_insulation: bool = False

    # ── Stage 예측 보조 손실 설정 ──────────────────────────────────────────────
    # subtask_loss_weight: stage classification 보조 손실 가중치.
    #   0.0이면 stage 예측 head가 있어도 손실에 기여하지 않는다.
    #   (기본 12-task baseline은 tokenized_prompt=[task_id] 1개뿐이므로
    #    stage 분기가 실행되지 않아 이 값과 관계없이 stage 손실이 0이다.)
    subtask_loss_weight: float = 0.0

    # ── Inpainting threshold 설정 ─────────────────────────────────────────────
    # time_threshold_inpaint: flow matching 추론 중 inpainting 강제 적용 구간의 하한.
    #   ODE를 t=1 → t=0 방향으로 적분할 때:
    #     t > threshold: 이전에 실행된 action(inpaint 구간)을 강제로 덮어쓴다.
    #     t ≤ threshold: 마지막 단계에서는 모델이 자유롭게 action을 조정하도록 둔다.
    #   0.3 → 전체 time [0, 1] 중 상위 70% 구간에서만 inpainting 강제 적용.
    time_threshold_inpaint: float = 0.3

    # ── Vision backbone freeze 설정 ───────────────────────────────────────────
    # freeze_vision_backbone: SigLIP vision encoder의 가중치를 학습 중 고정할지 여부.
    #   True(기본): 이미지 특징 추출 부분이 업데이트되지 않는다.
    #     → pi0 체크포인트의 강력한 vision 표현 능력을 보존한다.
    #     → 메모리와 연산량 절약.
    #   False: vision backbone도 fine-tuning 대상에 포함된다.
    #     → B1K 특화 시각 표현을 학습할 수 있지만 더 많은 데이터가 필요하다.
    freeze_vision_backbone: bool = True

    # ── 자동 초기화 ────────────────────────────────────────────────────────────
    def __post_init__(self):
        """task_embedding_dim이 None이면 backbone hidden width로 자동 설정한다.

        frozen dataclass이므로 object.__setattr__로 우회해서 값을 설정한다.
        이 방법이 공식 권장 패턴이며, frozen 규칙을 "의도적으로" 깨는 초기화 전용 우회이다.
        """
        if self.task_embedding_dim is None:
            # Gemma 설정에서 hidden state 차원(width)을 가져온다.
            # gemma_2b의 경우 width = 2048.
            paligemma_config = _gemma.get_config(self.paligemma_variant)
            object.__setattr__(self, "task_embedding_dim", paligemma_config.width)

    # ── FAST 유틸 메서드 ─────────────────────────────────────────────────────
    def get_fast_dim_ranges(self) -> list[tuple[int, int]]:
        """fast_encoded_dims 문자열을 파싱해 (시작, 끝) 범위 목록으로 반환한다.

        예:
            "0:6,7:23" → [(0, 6), (7, 23)]
            [(0, 6), (7, 23)] → [(0, 6), (7, 23)]  (이미 리스트면 그대로 반환)

        주의:
            - 이 설정은 12개 task subset 기준이므로,
              50-task 전용 PI_BEHAVIOR 체크포인트와는 task embedding shape 호환이 안 된다.
            - backbone은 pi0(pi05_base)에서 초기화하고,
              task embedding/head만 새로 학습한다.
        """
        if isinstance(self.fast_encoded_dims, str):
            ranges = []
            for range_str in self.fast_encoded_dims.split(','):
                start, end = map(int, range_str.strip().split(':'))
                ranges.append((start, end))
            return ranges
        return self.fast_encoded_dims

    def get_total_fast_dims(self) -> int:
        """FAST가 인코딩하는 action 차원의 총 개수를 반환한다.

        예:
            [(0, 6), (7, 23)] → (6-0) + (23-7) = 22 차원
        """
        return sum(end - start for start, end in self.get_fast_dim_ranges())

    # ── BaseModelConfig 오버라이드 ───────────────────────────────────────────
    @property
    @override
    def model_type(self) -> str:
        """모델 식별자 문자열.

        체크포인트 저장/로딩 시 어떤 모델 클래스를 써야 하는지 판별하는 데 사용된다.
        weight_loaders.py에서 'task_embeddings'가 있으면 PI_BEHAVIOR 체크포인트,
        없으면 Pi0 체크포인트로 판단한다.
        """
        return "pi_behavior"

    @override
    def create(self, rng: at.KeyArrayLike) -> "PiBehavior":
        """이 설정으로 PiBehavior 모델 인스턴스를 만든다.

        함수 내부에서 import하는 이유: pi_behavior.py가 이 파일을 import하고 있어서
        모듈 최상단에서 import하면 순환 참조(circular import)가 발생하기 때문이다.
        """
        from b1k.models.pi_behavior import PiBehavior
        return PiBehavior(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple["Observation", _model.Actions]:
        """모델 forward pass의 입력 shape/dtype spec을 반환한다.

        JAX의 jit/XLA 컴파일 전에 입력 shape를 미리 선언하는 용도이다.
        실제 데이터가 아니라 ShapeDtypeStruct(껍데기 배열)를 반환하므로
        실제 메모리를 차지하지 않는다.

        반환:
            (Observation spec, action spec)
            이미지 3종 (base, left_wrist, right_wrist): [B, 224, 224, 3] float32
            state: [B, action_dim] float32
            tokenized_prompt: [B, 2] int32
              → 기본 경로: [로컬 task_id, stage_id]
              → 단순 경로: [로컬 task_id] (shape[1]=1)
            action: [B, action_horizon, action_dim] float32
        """
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            obs_kwargs = {
                "images": {
                    # 로봇 base 카메라 (정면 전경 뷰)
                    "base_0_rgb": image_spec,
                    # 왼손 손목 카메라 (근접 조작 뷰)
                    "left_wrist_0_rgb": image_spec,
                    # 오른손 손목 카메라 (근접 조작 뷰)
                    "right_wrist_0_rgb": image_spec,
                },
                "image_masks": {
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                # 로봇 고유감각(proprioception): 관절 각도, 속도 등.
                # action과 차원(action_dim)을 맞춰 둔다.
                "state": jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                # tokenized_prompt: [로컬 task_id, stage_id].
                # 기본 경로(12-task baseline)에서는 stage_id는 무시된다.
                "tokenized_prompt": jax.ShapeDtypeStruct([batch_size, 2], jnp.int32),
                "tokenized_prompt_mask": jax.ShapeDtypeStruct([batch_size, 2], bool),
            }

            # FAST 보조 학습을 켰을 때만 fast 관련 필드를 spec에 추가한다.
            # 꺼져 있을 때 필드가 없어야 데이터 로더에서 불필요한 전처리를 하지 않는다.
            if self.use_fast_auxiliary:
                obs_kwargs["fast_tokens"] = jax.ShapeDtypeStruct(
                    [batch_size, self.max_fast_tokens], jnp.int32
                )
                obs_kwargs["fast_token_mask"] = jax.ShapeDtypeStruct(
                    [batch_size, self.max_fast_tokens], bool
                )

            observation_spec = Observation(**obs_kwargs)
            action_spec = jax.ShapeDtypeStruct(
                [batch_size, self.action_horizon, self.action_dim], jnp.float32
            )
            return observation_spec, action_spec
