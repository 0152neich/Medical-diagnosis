# Disease Predict System

## Danh mục

1. [Hướng dẫn cài đặt](#Hướng-dẫn-cài-đặt)
2. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)

## Hướng dẫn cài đặt

- Bước 1: clone repository về máy tính cá nhân
```
git clone https://github.com/0152neich/Medical-diagnosis.git
```
- Bước 2: Tải file .csv tại [data](https://drive.google.com/drive/folders/1y4WYq7rJwp46HFFculRJDh89xCRufsZX?usp=drive_link), sau đó lưu vào folder data

- Bước 3: Cài đặt dependencies
```
pip install -r requirements.txt
```

## Hướng dẫn sử dụng
- Bước 1: Chạy train model
```
python -m src.train
```
- Bước 2: Chạy streamlit
```
streamlit run app.py
```