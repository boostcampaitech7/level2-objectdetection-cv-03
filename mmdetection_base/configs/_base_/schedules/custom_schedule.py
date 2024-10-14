# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',          # StepLR 정책 사용
    warmup='linear',         # 학습 초기에 학습률을 선형적으로 증가
    warmup_iters=500,        # 500 iteration 동안 warmup 적용
    warmup_ratio=0.001,      # 초기 학습률은 최종 학습률의 0.1%에서 시작
    step=[8, 11]             # 8, 11 epoch에서 학습률 감소
)

runner = dict(type='EpochBasedRunner', max_epochs=12)
