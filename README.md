# pet_disease_detect

본 프로젝트는 반려동물 질병진단 데이터의 품질을 검증하는 프로젝트로 널리 알려진 Semantic Segmentation Model(Deeplabv3+)를 이용하여 테스트 성능을 점검하는 방식으로 진행되었다.

특이한 점은 주어진 평가지표가 보편적인 IoU, Dice가 아니라는 점이다.
IoU가 일정 수준(0.5 혹은 0.25)를 넘으면 True Positive, 질병 예측 자체를 잘못하는 경우 False Posivie 등으로 분류하는 방식을 취했다.

모델 개발 과정과 평가지표에 관한 자세한 내용은 "반려동물_질병진단_세그멘테이션모델리뷰.pdf" 파일에 정리했습니다.
