
import numpy as np
from collections import OrderedDict

# OmniGibson 없이도 proprioception 인덱스를 로컬에서 바로 참조하기 위한 테이블
# 기존에는 omnigibson.learning.utils.eval_utils의 PROPRIOCEPTION_INDICES를 사용했음
PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            # 좌/우 팔 관절 위치
            "arm_left_qpos": np.s_[158:165],
            "arm_right_qpos": np.s_[197:204],

            # 좌/우 그리퍼 관절 위치
            "gripper_left_qpos": np.s_[193:195],
            "gripper_right_qpos": np.s_[232:234],

            # 몸통 관절 위치, base velocity
            "trunk_qpos": np.s_[236:240],
            "base_qvel": np.s_[253:256],
        }
    ),
}