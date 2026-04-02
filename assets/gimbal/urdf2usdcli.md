./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/sz/code/rl/target_aiming/assets/gimbal/gimbal.urdf \
  /home/sz/code/rl/target_aiming/assets/gimbal/gimbal.usd \
  --merge-joints \
  --joint-stiffness 0.0 \
  --joint-damping 0.0 \
  --joint-target-type none \
  --fix-base \
