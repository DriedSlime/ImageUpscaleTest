# Real-ESRGAN 기반 이미지 업스케일링 프로그램

이 프로젝트는 [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)을 이용해 저해상도 이미지를 고해상도로 업스케일링하는 Python 프로그램입니다.

---

## ✨ 기능

- 저해상도 이미지 2배, 4배, 8배 업스케일링
- GPU(CUDA) 가속 지원
- 간단한 코드로 이미지 업스케일 가능

---

## 📦 설치 방법

### 1. 필수 라이브러리 설치
```bash
pip install torch torchvision
pip install realesrgan
