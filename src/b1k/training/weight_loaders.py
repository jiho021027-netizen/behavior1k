"""Pi0 체크포인트에서 PiBehavior 모델로 가중치를 로드하는 모듈 (weight_loaders.py)

이 파일의 역할:
    PiBehavior 모델은 두 종류의 체크포인트를 다룬다.

    1. Pi0 체크포인트:
        - 원래 pi0 / pi05_base 사전 학습 가중치.
        - task_embeddings, stage head 등 B1K 전용 파라미터가 없다.
        - 로딩 방식: backbone(PaliGemma/SigLIP/ActionExpert) 가중치만 가져오고,
                     B1K 신규 파라미터는 랜덤 초기화된 값을 유지한다.

    2. PI_BEHAVIOR 체크포인트:
        - B1K fine-tuning 중간 저장된 체크포인트.
        - task_embeddings 등 모든 파라미터가 포함되어 있다.
        - 로딩 방식: 모든 가중치를 체크포인트에서 그대로 가져온다.

사용 예:
    config = TrainConfig(
        ...
        weight_loader=PiBehaviorWeightLoader(
            params_path="gs://my-bucket/pi05_base_checkpoint"
        ),
    )

Reference:
    https://github.com/Physical-Intelligence/openpi
"""

import dataclasses
import logging
import re

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download

# openpi의 기본 weight loader 클래스들을 재수출(re-export)한다.
# 외부 코드에서 이 모듈만 import해도 기본 loader들을 쓸 수 있다.
from openpi.training.weight_loaders import (
    WeightLoader,         # 모든 loader의 추상 기반 클래스
    NoOpWeightLoader,     # 아무것도 로드하지 않는 더미 loader
    CheckpointWeightLoader,  # 일반 체크포인트 loader
    _merge_params,        # 두 파라미터 트리를 병합하는 내부 유틸 함수
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PiBehaviorWeightLoader(WeightLoader):
    """PiBehavior 모델 전용 체크포인트 로더.

    Pi0 체크포인트와 PI_BEHAVIOR 체크포인트를 자동으로 구별해서 로드한다.

    자동 감지 방법:
        - 체크포인트에 'task_embeddings' 키가 있으면 PI_BEHAVIOR 체크포인트로 판단.
        - 없으면 Pi0 체크포인트로 판단.

    PI_BEHAVIOR 체크포인트 처리:
        - 모든 가중치를 체크포인트에서 가져온다.
        - shape 검증을 위해 _merge_params를 빈 missing_regex로 호출한다.

    Pi0 체크포인트 처리:
        - backbone 가중치(PaliGemma, SigLIP, ActionExpert)는 체크포인트에서 가져온다.
        - B1K 신규 파라미터(task_embeddings 등)는 missing_regex에 매칭되어
          랜덤 초기화된 값(params)이 유지된다.

    Attributes:
        params_path: 체크포인트 디렉토리 경로 또는 GCS URI.
            예: "gs://my-bucket/checkpoints/pi05_base"
                "/home/user/checkpoints/pi05_base"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        """체크포인트를 로드해서 모델 파라미터를 초기화한다.

        처리 흐름:
            1. 체크포인트 경로가 GCS URI이면 로컬로 다운로드한다.
            2. PyTreeCheckpointer로 체크포인트를 열어 파라미터를 읽는다.
            3. 일부 체크포인트는 'params' 키 아래 중첩되어 있으므로 unwrap한다.
            4. nnx.State 형식의 체크포인트는 각 값 끝에 'value' 키가 있으므로 제거한다.
            5. 체크포인트 종류를 감지한다.
            6. 종류에 따라 적절한 방식으로 파라미터를 병합한다.

        Args:
            params: 모델이 랜덤 초기화한 파라미터 트리.
                Pi0 체크포인트를 로드할 때 B1K 신규 파라미터의 초기값으로 사용된다.

        Returns:
            로드/병합된 파라미터 트리. shape는 params와 동일해야 한다.

        Raises:
            orbax.checkpoint.Error: 체크포인트 파일이 손상되었거나 형식이 맞지 않을 때.
            ValueError: _merge_params에서 shape 불일치가 감지될 때.
        """
        # GCS URI(gs://...)나 HTTP URL이면 로컬 캐시로 다운로드한다.
        # 이미 로컬 경로이면 그대로 사용한다.
        params_path = download.maybe_download(self.params_path)
        logger.info(f"체크포인트 경로: {params_path}")

        # PyTreeCheckpointer: Orbax의 범용 체크포인트 열기 도구.
        # 구형/신형 체크포인트 형식을 모두 처리할 수 있다.
        with ocp.PyTreeCheckpointer() as ckptr:
            restored = ckptr.restore(params_path)

        # 일부 체크포인트는 최상위에 'params' 키를 가진 중첩 구조이다.
        # 예: {'params': {'PaliGemma': {...}, 'action_in_proj': {...}}}
        # 이 경우 내부 파라미터 딕셔너리만 추출한다.
        if isinstance(restored, dict) and "params" in restored:
            loaded_params = restored["params"]
            logger.debug("'params' 키를 unwrap했습니다.")
        else:
            loaded_params = restored

        # flax.nnx.State 형식으로 저장된 체크포인트는
        # 각 파라미터 값이 {'value': actual_array} 형태로 감싸져 있다.
        # 이 'value' 레이어를 제거해서 단순한 배열 트리로 만든다.
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            # 모든 말단 키가 "value"이면 nnx.State 형식이다.
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)
            logger.debug("nnx.State 형식 감지: 'value' 키를 제거했습니다.")

        # ── 체크포인트 종류 감지 ────────────────────────────────────────────
        # 'task_embeddings' 키의 존재 여부로 PI_BEHAVIOR vs Pi0를 구별한다.
        # Pi0 체크포인트에는 task_embeddings가 없으므로 이 방법이 확실하다.
        has_task_embeddings = 'task_embeddings' in loaded_params

        if has_task_embeddings:
            # ── PI_BEHAVIOR 체크포인트 처리 ───────────────────────────────
            # 모든 가중치가 있으므로 체크포인트에서 그대로 가져온다.
            # missing_regex="^$"는 어떤 키도 매칭하지 않는 정규식이므로
            # "누락 키를 허용하지 않음"을 의미한다.
            # → shape 불일치가 있으면 _merge_params에서 에러가 발생한다.
            logging.info(
                "PI_BEHAVIOR 체크포인트 감지: 모든 가중치를 체크포인트에서 로드합니다."
            )
            return _merge_params(loaded_params, params, missing_regex="^$")

        else:
            # ── Pi0 체크포인트 처리 ────────────────────────────────────────
            # Pi0 체크포인트에는 B1K 전용 파라미터가 없다.
            # missing_regex에 매칭되는 파라미터는 체크포인트 대신
            # 랜덤 초기화된 값(params)을 유지한다.
            logging.info(
                "Pi0 체크포인트 감지: "
                "backbone 가중치는 체크포인트에서 로드하고, "
                "B1K 신규 파라미터는 랜덤 초기화값을 유지합니다."
            )

            # missing_regex: 이 정규식에 매칭되는 파라미터 경로는
            # 체크포인트에 없는 것으로 처리해서 랜덤 초기화값을 유지한다.
            #
            # 각 패턴 설명:
            # task_embeddings:     12개 태스크 embedding 테이블 (12 × 2048).
            #                      Pi0의 50-task embedding과 shape가 달라 로드 불가.
            # stage_pred_from_vlm: VLM 출력 → stage 분류 head. Pi0에 없음.
            # task_stage_embeddings: (task, stage) 쌍별 embedding 테이블. Pi0에 없음.
            # gate_sincos:         subtask 정보 fusion gate. Pi0에 없음.
            # gate_task_stage:     task-stage embedding gate. Pi0에 없음.
            # gate_task:           task embedding gate. Pi0에 없음.
            # fusion_layer*:       task + subtask 정보 fusion MLP. Pi0에 없음.
            # stage_projection:    stage 표현 → task 차원 투영. Pi0에 없음.
            # task_subtask_fusion: (이전 버전 호환용) fusion 관련 구 파라미터명.
            # fast_token_embedding: FAST 토큰 embedding 테이블. Pi0에 없음.
            # fast_token_proj:     FAST 토큰 예측 head. Pi0에 없음.
            # kv_transform:        KVCacheTransform 가중치. Pi0에 없음.
            missing_regex = (
                ".*task_embeddings.*|"
                ".*stage_pred_from_vlm.*|"
                ".*task_stage_embeddings.*|"
                ".*gate_sincos.*|"
                ".*gate_task_stage.*|"
                ".*gate_task.*|"
                ".*fusion_layer.*|"
                ".*stage_projection.*|"
                ".*task_subtask_fusion.*|"
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*"
            )

            # _merge_params(loaded, init, missing_regex):
            #   - loaded에 있는 파라미터: loaded에서 가져온다.
            #   - loaded에 없고 missing_regex에 매칭되는 파라미터: init(랜덤 초기화)에서 가져온다.
            #   - loaded에 없고 missing_regex에도 매칭 안 되는 파라미터: 에러 발생.
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
