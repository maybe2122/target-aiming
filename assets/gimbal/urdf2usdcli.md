./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/maybe/code/rl/target-aiming/assets/gimbal/gimbal.urdf \
  /home/maybe/code/rl/target-aiming/assets/gimbal/gimbal.usd \
  --merge-joints \
  --joint-stiffness 0.0 \
  --joint-damping 0.0 \
  --joint-target-type none \
  --fix-base \
